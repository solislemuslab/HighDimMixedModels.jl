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
- act_num :: Integer between 1 and 5. We will only update all fixed effect parameters every act_num iterations. Otherwise, we update only the parameters in thea current active set.
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
@with_kw mutable struct Control
    tol::Float64  = 1e-4
    seed::Int = 770
    trace::Int = 2
    max_iter::Int = 1000
    max_armijo::Int = 20
    act_num::Int = 5
    ainit::Float64 = 1.0
    δ::Float64 = 0.1
    ρ::Float64 = 0.001
    γ::Float64 = 0.0
    lower::Float64 = 1e-6
    upper::Float64 = 1e8
    var_int::Tuple{Float64, Float64} = (0, 100)
    cov_int::Tuple{Float64, Float64} = (-50, 50)
    optimize_method::Symbol = :Brent
    thres::Float64 = 1e-4
end


"""
Fits penalized linear mixed effect model 

ARGUMENTS
Positional: 
- X :: Low dimensional design matrix for unpenalized fixed effects (assumed to include column of ones) (REQUIRED)
- G :: High dimensional design matrix for penalized fixed effects (assumed to not include column of ones) (REQUIRE)
- y :: Vector of responses (REQUIRED)
- grp :: Vector of strings of same length as y assigning each observation to a particular group (REQUIRED)
- Z :: Design matrix for random effects (optional, default is all columns of X) (assumed to include column of ones) (REQUIRED)
NOTE: Z is not expected to be given in block diagonal form. It should be a vertical stack of subject design matrices Z₁, Z₂, ...

Keyword:
- standardize :: boolean (default false), whether to standardize design matrices before performing algorithm. 
- penalty :: One of "scad" (default) or "lasso"
- λ :: Positive regularizing penalty (default is 10.0)
- scada :: Extra tuning parameter for the SCAD penalty (default is 3.7, ignored if penalty is "lasso"
- wts :: Vector of length number of penalized coefficients. Strength of penalty on covariate j is λ/wⱼ (Default is vector of 1's)
- init_coef :: Tuple of form (β, L, σ²) giving initial values for parameters. If unspecified, then inital values for parameters are 
calculated as follows: first, cross-validated LASSO that ignores grouping structure is performed to obtain initial estimates of the 
fixed effect parameters. Then, the random effect parameters are initialized as MLEs assuming the LASSO estimates are true fixed effect parameters.
- ψstr :: One of "diag" (default), "ident", or "sym", specifying covariance structure of random effects 
- control :: Struct with fields for hyperparameters of the algorithm 

OUTPUT
- Fitted model
"""
function lmmlasso(X::Matrix{Float64}, G::Matrix{Float64}, y::Vector{Float64}, grp::Vector{String}, Z::Matrix{Float64}; 
    standardize = false, penalty::String="scad", λ::Float64=10.0, scada::Float64=3.7, wts::Vector{Float64}=fill(1, size(G, 2)), 
    init_coef::Union{Vector, Nothing}=nothing, ψstr::String="diag", control=Control()) 


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
    @assert X[:,1] == ones(N) "First column of X must be all ones"
    @assert Z[:,1] == ones(N) "First column of Z must be all ones"

    #Check whether columns of Z are a subset of the columns of X, or is this not necessary?
    
    #Penalty related checks
    @assert penalty in ["scad", "lasso"] "penalty must be one of \"scad\" or \"lasso\""
    @assert λ > 0 "λ is regularization parameter, must be positive"
    @assert all(wts .> 0) "Weights must be positive"
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

    #Standardize design matrices if so desired
    if standardize
        XGor = copy(XG)
        meansx = mean(XG[:,Not(1)], dims=1)
        sdsx = std(XG[:,Not(1)], dims=1)
        XG = (XG[:,Not(1)] .- means) ./ sds
        XG = [fill(1, N) XG]

        Zor = copy(Z)
        meansz = mean(Z[:,Not(1)], dims=1)
        sdsz = std(Z[:,Not(1)], dims=1)
        Z = (Z[:,Not(1)] .- meansz) ./ sdsz
        Z = [fill(1, N) Z]
    end
    
    #Get grouped data, i.e. lists of matrices/vectors
    Zgrp, XGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp .== group,:], XG[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(XGgrp, XGᵢ); push!(ygrp, yᵢ)
    end

    # --- Initialize parameters -------------------
    # ---------------------------------------------
    if init_coef === nothing
        #Initialize fixed effect parameters using standard, cross-validated Lasso which ignores random effects
        lassopath = fit(LassoModel, XG[:,Not(1)], y; select = MinCVmse(Kfold(N, 10)))
        βstart = coef(lassopath) #Fixed effects
        #Initialize covariance parameters
        Lstart, σ²start = cov_start(XGgrp, ygrp, Zgrp, βstart)
    else
        βstart, Lstart, σ²start = init_coef
    end

    if ψstr == "sym"
        Lstart = Matrix(Lstart*I(m))
    elseif ψstr == "diag"
        Lstart = fill(Lstart, m) 
    elseif ψstr != "ident"
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
    Liter = Lstart
    σ²iter = σ²start
    cov_iter = vcat(ndims(Liter) == 2 ? vec(Liter) : Liter, sqrt(σ²iter))
    fct_iter = fct_start
    
    convβ = norm(βiter)^2
    conv_cov = norm(cov_iter)^2 
    conv_fct = fct_iter

    do_all = false
    converged = 0
    counter_in = 0
    global counter = 0

    while (max(convβ, conv_cov, conv_fct) > control.tol || !do_all) && (counter < control.max_iter)
            
        counter += 1
        control.trace > 2 && println("Cost before iteration $counter: $fct_iter")

        if counter == control.max_iter 
            println("Maximum of $counter iterations reached")
            converged = 1
        end
        
        #Keep copy of variables that are being updated for convergence checking
        βold = copy(βiter)
        cov_old = copy(cov_iter)
        fct_old = fct_iter

        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"
        #See  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(βiter .!= 0)
        
        if counter_in == 0 || counter_in > control.act_num
            active_set = 1:(p+q)
            counter_in = 1
            do_all = true
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
                    βiter[j] = cut/hess[j]
                elseif penalty == "lasso" 
                    βiter[j] = soft_thresh(cut, λwtd[j])/hess[j]
                else #Scad penalty
                    βiter[j] = scad_solution(cut, hess[j], λwtd[j], scada)
                end 
            else #Must actually perform Armijo line search 
                fct_iter, converged = armijo!(XGgrp, ygrp, invVgrp, βiter, j, p, cut, hess_untrunc[j],
                hess[j], penalty, λwtd, scada, fct_iter, converged, control)
            end 
        end

        #---Optimization with respect to covariance parameters ----------------------------
        #------------------------------------------------------------------------------------

        if ψstr == "ident"
            #println(counter)
            #println(Liter)
            #println(σ²iter)
            Liter = L_ident_update(XGgrp, ygrp, Zgrp, βiter, σ²iter, control.var_int, control.thres)
        elseif ψstr == "diag"
            for s in 1:m 
                L_diag_update!(Liter, XGgrp, ygrp, Zgrp, βiter, σ²iter, s, control.var_int, control.thres)
            end
        else  #ψstr == "sym"
            for i in 2:m
                for j in 1:(i-1)
                    L_sym_update!(Liter, XGgrp, ygrp, Zgrp, βiter, σ²iter, (i,j), control.var_int, control.cov_int, control.thres)
                end
            end
        end

        # Optimization of σ²
        σ²iter = σ²update(XGgrp, ygrp, Zgrp, βiter, Liter, control.var_int)
        
        #Vector of variance/covariance parameters
        cov_iter = vcat(ndims(Liter) == 2 ? vec(Liter) : Liter, sqrt(σ²iter))

        #Calculate new inverse variances 
        Vgrp = var_y(Liter, Zgrp, σ²iter)
        invVgrp = [inv(V) for V in Vgrp]

        #Calculate new objective function
        neglike = get_negll(invVgrp, ygrp, XGgrp, βiter)
        fct_iter = get_cost(neglike, βiter[(p+1):end], penalty, λwtd[(p+1):end], scada)

        #Check convergence
        convβ = norm(βiter-βold)/(1+norm(βiter))
        conv_cov = norm(cov_iter-cov_old)/(1+norm(cov_iter))
        conv_fct = abs(fct_iter-fct_old)/(1+abs(fct_iter))

        if max(convβ, conv_cov, conv_fct) <= control.tol
            counter_in = 0 #Update all entries next iteration and if parameters still doesn't change, then call it quits
        end

    end

    #Get coefficients and design matrices on the original scale 
    if (standardize)
        βiter[Not(1)] = βiter[Not(1)] ./ sdsx
        βiter[1] = βiter[1] - sum(meansx .* βiter[Not(1)])

        XG = XGor
        Z = Zor

        for group in unique(grp)
            Zᵢ, XGᵢ = Z[grp .== group,:], XG[grp .== group,:]
            push!(Zgrp, Zᵢ); push!(XGgrp, XGᵢ)
        end
    end


    #Prediction of random effects


    #Final calculations


    #Form output and return
    return βiter, Liter, σ²iter, counter

end

end
