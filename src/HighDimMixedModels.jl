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
    ainit::Float64 = 1.0
    δ::Float64 = 0.1
    ρ::Float64 = 0.001
    γ::Float64 = 0.0
    lower::Float64 = 1e-6
    upper::Float64 = 1e8
    var_int::Tuple{Float64, Float64} = (0, 1000)
    cov_int::Tuple{Float64, Float64} = (-500, 500)
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
- Z :: Design matrix for random effects (optional, default is all columns of X) (assumed to include column of ones) (REQUIRED)
NOTE: Z is not expected to be given in block diagonal form. It should be a vertical stack of subject design matrices Z₁, Z₂, ...

Keyword:
- penalty :: One of "scad" (default) or "lasso"
- λ :: Positive regularizing penalty (default is 10.0)
- scada :: Extra tuning parameter for the SCAD penalty (default is 3.7, ignored if penalty is "lasso"
- wts :: Vector of length number of penalized coefficients. Strength of penalty on covariate j is λ/wⱼ (Default is vector of 1's)
- init_coef :: Tuple of form (β, L, σ²) giving initial values for parameters. If unspecified, then inital values for parameters are 
calculated as follows: first, cross-validated LASSO that ignores grouping structure is performed to obtain initial estimates of the 
fixed effect parameters. Then, the random effect parameters are initialized as MLEs assuming the LASSO estimates are true fixed effect parameters.
- Ψstr :: One of "diag" (default), "ident", or "sym", specifying covariance structure of random effects 
- control :: Struct with fields for hyperparameters of the algorithm 

OUTPUT
- Fitted model
"""
function lmmlasso(X::Matrix{Real}, G::Matrix{Real}, y::Vector{Real}, grp::Vector{String}, Z=X::Matrix{Real}; 
    penalty::String="scad", λ::Float64=10.0, scada::Float64=3.7, wts::Vector{Real}=fill(1, size(G, 2)), 
    init_coef::Union{Vector, Nothing}=nothing, Ψstr::String="diag", control=Control()) 


    ##Get dimensions
    N = length(y) # Total number of observations
    q = size(G, 2) # Number of penalized covariates 
    p = size(X, 2) # Number of unpenalized covariates 
    m = size(Z, 2) # Number of covariates associated with random effects 
    groups = unique(grp)
    g = length(groups) #Number of groups



    # --- Introductory checks ---------------------
    # ---------------------------------------------

    #Dimension compatability checks
    @assert size(G, 1) == N "G and y incompatable dimension"
    @assert size(X, 1) == N "X and y incompatable dimension"
    @assert size(Z, 1) == N "Z and y incompatable dimension"
    @assert length(grp) == N "grp and y incompatable dimension"
    @assert g > 1 "Only one group, no covariance parameters can be estimated"

    #Intercept included checks
    @assert X[:,1] == ones(n_tot) "First column of X must be all ones"
    @assert Z[:,1] == ones(n_tot) "First column of Z must be all ones"

    #Check whether columns of Z are a subset of the columns of X, or is this not necessary?
    
    #Penalty related checks
    @assert penalty in ["scad", "lasso"] "penalty must be one of \"scad\" or \"lasso\""
    @assert λ > 0 "λ is regularization parameter, must be positive"
    @assert wts .> 0 "Weights must be positive"
    λwtd = [zeros(p); λ./wts] 
    @assert ψstr in ["ident", "diag", "sym"] "ψstr must be one of \"ident\", \"diag\", or \"sym\""
    @assert control.optimize_method in [:Brent, :GoldenSection] "Control.optimize_method must be one of :Brent or :GoldenSection"
    
    control::HighDimMixedModels.Control #Checks control hyper parameters

    #Check that initial coefficients, if provided, make sense
    if init_coef !== nothing
        @assert length(init_coef) == 3 "init_coef must be of length 3"
    end


    # --- Introductory allocations ----------------
    # ---------------------------------------------

    #Merge unpenalized and penalized design matrices
    XG = [X G] 
    
    #Get grouped data, i.e. lists of matrices/vectors
    Zgrp, XGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp .== group,:], XG[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(XGgrp, XGᵢ); push(ygrp, yᵢ)
    end

    # --- Initialize parameters -------------------
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
        Lstart = LowerTriangular(Lstart*I(m))
    elseif Ψstr == "diag"
        Lstart = fill(L, m) 
    elseif Ψstr != "ident"
        error("ψstr must be one of \"sym\", \"diag\", or \"ident\"")
    end

    # --- Calculate cost for the starting values ---
    # ---------------------------------------------
    Vgrp = var_y(Lstart, Zgrp, σ²start)
    invVgrp = [inv(V) for V in Vgrp]
    neglike_start = get_negll(invVgrp, ygrp, XGgrp, βstart)
    fct_start = get_cost(neglike_start, βstart[(p+1):end], penalty, λwtd[(p+1):end], scada)
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

    converged = 0
    counter_in = 0
    counter = 0

    while (converged == false) && (control.max_iter > counter)
        
        counter += 1
        counter == control.max_iter && println("Maximum of $counter iterations reached")

        #Variables that are being updated
        βold = copy(βiter)
        Lold = copy(Liter)
        σ²old = σ²iter
        fct_old = fct_iter


        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"--
        #see  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(β .!= 0)
        
        if counter_in == 0 || counter_in > control.act_min 
            active_set = 1:(p+q)
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

            if hess[j] == hess_untrunc[j] #Outcome of Armijo rule can be computed analytically
                if j in 1:p
                    βiter[j] = cut/hess_diag[j]
                elseif penalty == "lasso" 
                    βiter[j] = soft_thresh(cut, λwtd[j])/hess[j]
                else #Scad penalty
                    βiter[j] = scad_solution(cut, hess[j], λwtd[j], scada)
                end 
            else #Must actually perform Armijo line search 
                fct_iter, converged = armijo!(XGgrp, ygrp, invVgrp, βiter, j, p, cut, hess_untrunc[j],
                hess[j], penalty, λwtd, a, fct_iter, converged, control)
            end 
        end

        #---Optimization with respect to covariance parameters ----------------------------
        #------------------------------------------------------------------------------------

        # calculations before the covariance optimization
        active_set = findall(fpars .!= 0)
        resgrp = resid(XaugGgrp, ygrp, fpars, active_set)
        
        # Optimization of L
        if ψstr == :ident
            Liter = L_ident_update(XGgrp, ygrp, Zgrp, βiter, σ²iter, control.cov_int, control.thres)
        elseif ψstr == :diag
            foreach( s -> L_diag_update!(Liter, XGgrp, ygrp, Zgrp, β, σ²iter, s, control.var_int, control.thres), 1:m)
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
        neglike = get_negll(invVgrp, ygrp, XaugGgrp, fpars)
        #cost = neglike + λ.*abs.(β)

        #Compare to previous cost function...

    end
end


end
