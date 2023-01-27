module HighDimMixedModels

export lmmlasso
export Control

include("helpers.jl")
using Parameters #Supplies macro for structs with default field values

"""
Algorithm Hyper-parameters

- tol :: Convergence tolerance
- seed :: Random seed for cross validation for estimating initial fixed effect parameters using Lasso
- trace :: Integer. 1 prints no output, 2 prints warnings, and 3 prints the objective function values during the algorithm and warnings
- max_iter :: Integer. Maximum number of iterations
- max_armijo :: Integer. Maximum number of steps in Armijo rule algorithm. If the maximum is reached, algorithm doesn't update current coordinate and proceeds to the next coordinate
- act_number :: Integer between 1 and 5. We will only update all fixed effect parameters every act_number iterations. Otherwise, we update only the parameters in thea current active set.
- a₀ :: a₀ in the Armijo step. See Schelldorfer et al. (2010)
- δ :: δ in the Armijo step. See Schelldorfer et al. (2010)
- ρ :: ρ in the Armijo step. See Schelldorfer et al. (2010)
- γ :: γ in the Armijo step. See Schelldorfer et al. (2010)
- lower :: Lower bound for the Hessian
- upper :: Upper bound for the Hessian
- var_int :: Tuple with bounds of interval on which to optimize the variance parameters used in `optimize` function. See Optim.jl in section "minimizing a univariate function on a bounded interval"
- cov_int :: Tuple with bounds of interval on which to optimize the covariance parameters used in `optimize` function. See Optim.jl in section "minimizing a univariate function on a bounded interval"
- optimize_method :: Symbol denoting method for performing the univariate optimization, either :Brent or :GoldenSection
- thres :: If variance or covariance parameter has smaller absolute value than `thres`, parameter is set to 0
"""
@with_kw struct Control
    tol::Float64  = 1e-4
    seed::Int = 770
    trace::Int = 2
    max_iter::Int = 1000
    max_armijo::Int = 20
    act_number::Int = 5
    a₀::Float64 = 1.0
    δ::Float64 = 0.1
    ρ::Float64 = 0.001
    γ::Float64 = 0.0
    lower::Float64 = 1e-6
    upper::Float64 = 1e8
    var_int::Tuple{Float64, Float64} = (0.0, 10.0)
    cov_int::Tuple{Float64, Float64} = (-5.0, 5.0)
    optimize_method::Symbol = :Brent
    thres::Float64 = 1e-4
end


"""
Fits penalized linear mixed effect model 

ARGUMENTS
Positional: 
- X :: Low dimensional design matrix for unpenalized fixed effects (assumed to include column of ones) (REQUIRED)
- G :: High dimensional design matrix for penalized fixed effects (not assumed to include column of ones) (REQUIRE)
- y :: Vector of responses (REQUIRED)
- grp :: Vector of strings of same length as y assigning each observation to a particular group (REQUIRED)
- Z :: Design matrix for random effects (optional, default is all columns of X)
Keyword:
- λ :: Positive regularizing penalty (optional, default is 10.0)
- weights :: Vector of length number of penalized coefficients. Strength of penalty of covariate j is λ/wⱼ   
- init_coef :: Tuple with three elements: starting value for fixed effects, random effect, and error variance parameters
- Ψstr :: One of "diag" (default), "ident", or "sym", specifying covariance structure of random effects 
- control :: Struct with fields for hyperparameters of the algorithm

OUTPUT
- Fitted model
"""
function lmmlasso(X::Matrix{Real}, G::Matrix{Real}, y::Vector{Real}, grp::Vector{String}, Z=X::Matrix{Real}; 
    λ::Float64=10.0, weights::Vector{Real}=fill(1, size(G, 2)), init_coef::Union{Vector, Nothing}=nothing, 
    Ψstr::String="diag", control=Control()) 

    # --- Introductory checks --- 
    # ---------------------------------------------
    N = length(y) # Total number of observations
    q = size(G, 2) # Number of penalized covariates 
    p = size(X, 2) # Number of unpenalized covariates 
    m = size(Z, 2) # Number of covariates associated with random effects 

    @assert size(G, 1) == N "G and y incompatable dimension"
    @assert size(X, 1) == N "X and y incompatable dimension"
    @assert size(Z, 1) == N "Z and y incompatable dimension"
    @assert length(grp) == N "grp and y incompatable dimension"

    @assert X[:,1] == ones(n_tot) "First column of X must be all ones"
    groups = unique(grp)
    g = length(groups) #Number of groups
    @assert g > 1 "Only one group, no covariance parameters can be estimated"
    
    @assert λ > 0 "λ is regularization parameter, must be positive"

    @assert ψstr in ["ident", "diag", "sym"] "ψstr must be one of \"ident\", \"diag\", or \"sym\""
    control::HighDimMixedModels.Control 
    @assert control.optimize_method in [:Brent, :GoldenSection] "Control.optimize_method must be one of :Brent or :GoldenSection"
    if init_coef !== nothing
        @assert length(init_coef) == 3 "init_coef must be of length 3"
    end

    # --- Intro allocations -----------------------
    # ---------------------------------------------
    XG = [X G] # Merging unpenalized and penalized design matrices
    
    #Grouped data
    Zgrp, XGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp .== group,:], XG[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(XGgrp, XGᵢ); push(ygrp, yᵢ)
    end
    
    # --- Initializing parameters ---
    # ---------------------------------------------
    if init_coef === nothing
        Random.seed!(control.seed)
        #Initialize fixed effect parameters using standard, cross-validated Lasso which ignores random effects
        lassopath = fit(LassoModel, XG[:,Not(1)], y; select = MinCVmse(Kfold(n_tot, 10)))
        βstart = coef(lassopath; select=MinCVmse(lassopath, 10)) #Fixed effects
        #Initialize covariance parameters
        Lstart, σ²start = cov_start(XGgrp, ygrp, Zgrp, fpars)
    else
        βstart, Lstart, σ²start = init_coef
    end

    if Ψstr == "sym"
        Lstart = LowerTriangular(L*I(m+1))
    elseif Ψstr == "diag"
        Lstart = fill(L, m+1) 
    elseif Ψstr != "ident"
        error("ψstr must be one of \"sym\", \"diag\", or \"ident\"")
    end

    # --- Calculate objective function for the starting values ---
    # ---------------------------------------------
    Vgrp = var_y(Lstart, Zgrp, σ²start)
    invVgrp = [inv(V) for V in Vgrp]
    neglike_start = negloglike(invVgrp, ygrp, XGgrp, βstart)
    fct_start = neglike_start + λ*norm(βstart[(p+1):end], 1)./weights
    control.trace > 2 && println("Cost at initialization: $fct_start")


    # --- Coordinate Gradient Descent -------------
    # ---------------------------------------------

    #Algorithm allocations
    βiter = βstart
    σ²iter = σ²start
    Liter = Lstart
    fct_iter = fct_start
    convβ = norm(βiter)^2
    conv_cov = norm(Liter)^2 +  σ²iter
    conv_fct = fct_iter

    converged = false
    counter_in = 0
    counter = 0

    while (converged == false) && (control.max_iter > counter)
        
        counter += 1
        counter == control.max_iter && println("Maximum of $counter iterations reached")

        #Variables that are being updated
        βold = βiter
        Lold = Liter
        σ²old = σ²iter 
        fct_old = fct_iter


        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"--
        #see  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(β .!= 0)
        
        if counter_in == 0 || counter_in > control.act_min 
            active_set = 1:(p+q+1)
            counter_in = 1
        else 
            counter_in += 1
        end

        hess = hessian_diag(XGgrp, invVgrp, active_set)
        hess_untrunc = copy(hess)
        hess[active_set] = min.(max.(hess[active_set], control.lower), control.upper)


        #Update fixed effect parameters that are in active_set
        for j in active_set 

            cut = special_quad(XGgrp, invVgrp, ygrp, βiter, j)

            if hess_diag[j] == hess_diag_untrunc[j] #Outcome of Armijo rule can be computed analytically
                if j in 1:p+1
                    fpar[j] = cut/hess_diag[j]
                else
                    fpar[j] = soft_thresh(cut, λ)/hess_diag[j]
                end 
            else #Must actually perform Armijo line search 
                arm = armijo(Xgrp, ygrp, Vgrp, fpars, j, cut, hess_diag_untrunc[j], hess_diag[j], cost, p, converged)
                fpars = arm.fpars
                cost = arm.cost
                converged = arm.converged
            end
            control.trace > 3 && println(cost) 
        end
        β = fpars[(p+2):end] #Pull out new penalized coefficients

        control.trace > 3 && println("------------------------")


        #---Optimization with respect to covariance parameters ----------------------------
        #------------------------------------------------------------------------------------

        # calculations before the covariance optimization
        active_set = findall(fpars .!= 0)
        resgrp = resid(XaugGgrp, ygrp, fpars, active_set)
        
        # Optimization of L
        if ψstr == :ident
            L = L_ident_update(XaugGgrp, ygrp, Zgrp, fpars, σ², control.var_int, control.thres)
        elseif ψstr == :diag
            foreach( s -> L_diag_update!(L, XaugGgrp, ygrp, Zgrp, fpars, σ², s, control.var_int, control.thres), 1:m)
        else  #ψstr == :sym
            tuples = collect(Iterators.product(1:m, 1:m)) #Matrix of tuples
            tuples = tuples[tril!(trues((m,m)), 0)] #Vector of tuples (i, j), where i≥j
            foreach( coords -> L_sym_update!(L, XaugGgrp, ygrp, Zgrp, fpars, σ², coords, control.var_int, control.cov_int, control.thres), 
                    tuples)
        end

        control.trace > 3 && println("------------------------")

        # Optimization of σ²
        σ² = σ²update(XaugGgrp, ygrp, Zgrp, fpars, L, control.var_int)
        control.trace > 3 && println("------------------------")

        #Calculate new cost function
        Vgrp = Vgrp(L, Zgrp, σ²)
        invVgrp = [inv(V) for V in Vgrp]
        neglike = negloglike(invVgrp, ygrp, XaugGgrp, fpars)
        cost = neglike + λ*norm(β, 1)

        #Compare to previous cost function...

    end
end


end
