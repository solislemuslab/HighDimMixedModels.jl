module HighDimMixedModels

# Dependencies
using Statistics
using LinearAlgebra
using Random
using Optim: Optim, optimize
using Parameters: @with_kw
using MLBase: Kfold
using Lasso: Lasso
using StatsBase: StatsBase

# Export the two structs and the main function defined in this package
export Control, HDMModel, hdmm

# Methods are defined for the functions from the StatsAPI in the file statsbase.jl
import StatsAPI: coef, coeftable, deviance, loglikelihood, nobs, residuals, fitted, aic, bic
export coef, coeftable, deviance, loglikelihood, nobs, residuals, fitted, aic, bic

include("helpers.jl")
include("structs.jl")
include("statsbase.jl")


"""
    hdmm(X::Matrix{<:Real}, G::Matrix{<:Real}, y::Vector{<:Real}, 
         grp::Vector{<:Union{String, Int64}}, Z::Matrix{<:Real}=X; 
         <keyword arguments>)

Fit a penalized linear mixed effect model using the coordinate gradient descent (CGD) 
algorithm and return a fitted model of type `HDMModel`. 

# Arguments
- `X`: Low dimensional (N by q) design matrix for unpenalized fixed effects (**first column must be all 1's to fit intercept**) 
- `G`: High dimensional (N by p) design matrix for penalized fixed effects (**should not include column of 1's**) 
- `y`: Length N response vector
- `grp`: Length N vector with group assignments of each observation 
- `Z=X`: Random effects design matrix (N by m), should contain some subset of the columns of `X` (defaults to equal `X`)
Keyword: 
- `penalty::String="scad"`: Either "scad" or "lasso"
- `standardize::Bool=true`: Whether to standardize the columns of `G` before fitting. The value of `λ` (and `wts`) should be chosen accordingly. Coefficient estimates are returned on the original scale.
- `λ::Real=10.0`: Positive number providing the regularization parameter for the penalty
- `wts::Union{Vector,Nothing}=nothing`: If specified, the penalty on covariate j will be λ/wⱼ, so this argument is useful if you want to penalize some covariates more than others. 
- `scada::Real=3.7`: Positive number providing the extra tuning parameter for the scad penalty (ignored for lasso)
- `max_active`::Real=length(y)/2: Maximum number of fixed effects estimated non-zero (defaults to half the total sample size)
- `ψstr::String="diag"`: One of "ident", "diag", or "sym", specifying the structure of the random effects' covariance matrix 
- `init_coef::Union{Tuple,Nothing} = nothing`: If specified, provides the initialization to the algorithm. See notes below for more details
- `control::Control = Control()`: Custom struct with hyperparameters of the CGD algorithm, defaults are in documentation of `Control` struct

# Notes
The initialization to the descent algorithm can be specified in the `init_coef` argument as a tuple of the form (β, L, σ²), where
- β is a vector of length p + q providing an initial estimate of the fixed effect coefficients
- L is the Cholesky factor of the random effect covariance matrix, and is represented as 
    - a scalar if ψstr="ident"
    - a vector of length m if ψstr="diag"
    - a lower triangular matrix of size m by m if ψstr="sym"
- σ² is a scalar providing an initial estimate of the noise variance
If the `init_coef` argument is not specified, we obtain initial parameter estimates in the following manner:
1. A LASSO that ignores random effects (with λ chosen using cross validation) is performed to estimate the fixed effect parameters. 
2. L, assumed temporarilly to be a scalar, and σ² are estimated to maximize the likelihood given these estimated fixed effect parameters.
3. If `ψstr` is "diag" or "sym", the scalar L is converted to a vector or matrix by repeating the scalar or filling the diagonal of a matrix with the scalar, respectively.
"""
function hdmm(
    X::Matrix{<:Real},
    G::Matrix{<:Real},
    y::Vector{<:Real},
    grp::Vector{<:Union{String,Int64}},
    Z::Matrix{<:Real} = X;
    penalty::String = "scad",
    standardize::Bool = true,
    λ::Real = 10.0,
    wts::Union{Vector,Nothing} = nothing,
    scada::Real = 3.7,
    max_active::Real = length(y)/2,
    ψstr::String = "diag",
    init_coef::Union{Tuple,Nothing} = nothing,
    control::Control = Control(),
)

    #Get dimensions
    N = length(y) # Total number of observations
    p = size(G, 2) # Number of penalized covariates 
    q = size(X, 2) # Number of unpenalized covariates 
    m = size(Z, 2) # Number of covariates associated with random effects (less than or equal to q) 
    groups = unique(grp)
    g = length(groups) #Number of groups


    # --- Introductory checks ---------------------
    # ---------------------------------------------

    #Dimension compatability checks
    size(G, 1) == N || throw(ArgumentError("G and y incompatable dimension"))
    size(X, 1) == N || throw(ArgumentError("X and y incompatable dimension"))
    size(Z, 1) == N || throw(ArgumentError("Z and y incompatable dimension"))
    length(grp) == N || throw(ArgumentError("grp and y incompatable dimension"))
    g > 1 || throw(ArgumentError("Only one group, no covariance parameters can be estimated"))

    #Intercept included check
    X[:, 1] == ones(N) || throw(ArgumentError("First column of X must be all ones"))
    Z_int = (Z[:, 1] == ones(N)) #Bool for whether Z includes intercept (i.e. whether we want random intercepts)

    #Penalty related checks
    penalty in ["scad", "lasso"] || throw(ArgumentError("penalty must be one of \"scad\" or \"lasso\""))
    λ > 0 || throw(ArgumentError("λ is regularization parameter, must be positive"))
    scada > 0 || throw(ArgumentError("scada is regularization parameter, must be positive"))
    wts = (wts === nothing ? fill(1, p) : wts)
    all(wts .> 0) || throw(ArgumentError("Weights must be positive"))
    length(wts) == p || throw(ArgumentError("Number of weights must equal number of penalized covariates"))
    # Create the vector of penalty parameters of length q + p, i.e. one for each fixed effect
    λwtd = [zeros(q); λ ./ wts]
    ψstr in ["ident", "diag", "sym"] || throw(ArgumentError("ψstr must be one of \"ident\", \"diag\", or \"sym\""))
    control.optimize_method in [:Brent, :GoldenSection] || throw(ArgumentError("Control.optimize_method must be one of :Brent or :GoldenSection"))
    # Warn about behavior of cost function when using SCAD penalty
    if control.trace && penalty == "scad"
        printstyled("""
                    Note that under SCAD penalty, the algorithm is not minimizing the original cost function, 
                    so objective won't necesarilly decrease monotonically.
                    """, color = :cyan, bold = true)
    end

    #Check that initial coefficients, if provided, make sense
    if init_coef !== nothing
        length(init_coef) == 3 || throw(ArgumentError("init_coef must be of length 3"))
        init_coef.β isa Vector{Real} || throw(ArgumentError("init_coef.β must be a vector of reals"))
        length(init_coef.β) == q + p || throw(ArgumentError("init_coef.β must be of length q + p"))
        if ψstr == "ident"
            init_coef.L isa Real || throw(ArgumentError("init_coef.L must be a real number"))
            init_coef.L > 0 || throw(ArgumentError("init_coef.L must be positive"))
        elseif ψstr == "diag"
            init_coef.L isa Vector{Real} || throw(ArgumentError("init_coef.L must be a vector of reals"))
            length(init_coef.L) == m || throw(ArgumentError("init_coef.L must be of length m"))
            all(init_coef.L .> 0) || throw(ArgumentError("entries of init_coef.L must be positive"))
        elseif ψstr == "sym"
            init_coef.L isa Matrix{Real} || throw(ArgumentError("init_coef.L must be a matrix of reals"))
            size(init_coef.L) == (m, m) || throw(ArgumentError("init_coef.L must be of size m by m"))
            all(diag(init_coef.L) .> 0) || throw(ArgumentError("diagonal entries of init_coef.L must be positive"))
            istril(init_coef.L) || throw(ArgumentError("init_coef.L must be lower triangular"))
        end
        init_coef.σ² isa Real || throw(ArgumentError("init_coef.σ² must be a real number"))
        init_coef.σ² > 0 || throw(ArgumentError("init_coef.σ² must be positive"))
    end

    # --- Introductory allocations ----------------
    # ---------------------------------------------

    #Standardize penalized design matrix G if so desired
    if standardize
        meansG = mean(G, dims = 1)
        sdsG = std(G, dims = 1)
        G = (G .- meansG) ./ sdsG
    end

    #Merge unpenalized and penalized design matrices
    XG = [X G]

    #Get grouped data, i.e. lists of matrices/vectors
    Zgrp, XGgrp, ygrp = Matrix{eltype(Z)}[], Matrix{eltype(XG)}[], Vector{eltype(y)}[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp.==group, :], XG[grp.==group, :], y[grp.==group]
        push!(Zgrp, Zᵢ)
        push!(XGgrp, XGᵢ)
        push!(ygrp, yᵢ)
    end

    # --- Initialize parameters -------------------
    # ---------------------------------------------
    if init_coef === nothing
        #Initialize fixed effect parameters using standard Lasso which ignores random effects
        # First select λ using cross-validation.
        # If error occurs, it's due to rank deficiency in X for given training set;
        # In this case, select λ based on BIC
        lassopath = try
            Lasso.fit(
                Lasso.LassoModel,
                XG[:, 2:end], #Function for LASSO fitting accepts design matrix without intercept column and assumes you want intercept fit
                y;
                maxncoef=max(2 * N, 2 * p), #See https://github.com/JuliaStats/Lasso.jl/issues/54
                penalty_factor=[zeros(q - 1); 1 ./ wts], # Don't penalize first q-1 covariates (q-1 because intercept has been removed)
                select=Lasso.MinCVmse(Kfold(N, 10)), # Use cross-validation to select λ
            )
        catch
            Lasso.fit(
                Lasso.LassoModel,
                XG[:, 2:end],
                y;
                maxncoef=max(2 * N, 2 * p),
                penalty_factor=[zeros(q - 1); 1 ./ wts],
                select=Lasso.MinBIC(),
            )
        end
        βstart = Lasso.coef(lassopath)
        #Initialize covariance parameters
        Lstart, σ²start = cov_start(XGgrp, ygrp, Zgrp, βstart)
        #Get L in form specified by ψstr
        if ψstr == "sym"
            Lstart = Matrix(Lstart * I(m))
        elseif ψstr == "diag"
            Lstart = fill(Lstart, m)
        end
    else
        βstart, Lstart, σ²start = init_coef
    end

    #Number of non-zeros in initial fixed-effect estimates 
    nz_start = sum(βstart .!= 0)

    # --- Calculate cost for the starting values ---
    # ---------------------------------------------
    invVgrp = Vector{Matrix{Float64}}(undef, g)
    invV!(invVgrp, Zgrp, Lstart, σ²start)
    neglike_start = get_negll(invVgrp, ygrp, XGgrp, βstart)
    fct_start = get_cost(neglike_start, βstart[(q+1):end], penalty, λwtd[(q+1):end], scada)
    control.trace && println("Objective at initialization: $fct_start")

    # --- Coordinate Gradient Descent -------------
    # ---------------------------------------------

    #Algorithm allocations
    βiter = copy(βstart)
    res_iter = y - XG * βiter
    Liter = copy(Lstart)
    σ²iter = σ²start
    Lvec_iter = ndims(Liter) == 2 ? vec(Liter) : Liter #L as vector if ψstr == "sym"
    varp_iter = vcat(Lvec_iter, sqrt(σ²iter)) #Vector of covariance parameters
    neglike_iter = neglike_start
    fct_iter = fct_start
    hess = zeros(q + p) #Container for future calculations
    mat = zeros(g, q + p) #Container for future calculation

    convβ = norm(βiter)^2
    conv_varp = norm(varp_iter)^2
    conv_fct = fct_iter

    stopped = false #Make true if we start interpolating the data
    do_all = false
    counter_in = 0
    counter = 0
    num_arm = 0 #Goes up by 1 every time Armijo needs to be performed (cannot be computed analytically)
    arm_con = 0 #Goes up by 1 every time an Armijo step fails to converge

    while (max(convβ, conv_varp, conv_fct) > control.tol || !do_all) &&
        (counter < control.max_iter)

        counter += 1

        #Keep copy of variables that are being updated for convergence checking
        βold = copy(βiter)
        varp_old = copy(varp_iter)
        fct_old = fct_iter

        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"
        #See  page 53 of lmmlasso dissertation 
        active_set = findall(βiter .!= 0)
        control.trace && println("And size of active set is: $(length(active_set))")
        # If the active set is larger than half the total sample size and we've iterated at least once,
        # that means we're converging towards a non-sparse solution 
        # Tell user to increase λ
        if length(active_set) > max_active && λ > 0 && counter >= 2
            stopped = true
            break
        end

        # We make the active set include all covariates every act_num iterations
        if counter_in == 0 || counter_in >= control.act_num
            active_set = 1:(p+q)
            counter_in = 1
            do_all = true
        else
            do_all = false
            counter_in += 1
        end

        #Calculate the Hessian for the coordinates of the active set
        hessian_diag!(XGgrp, invVgrp, active_set, hess, mat[:, active_set])
        hess_untrunc = copy(hess)
        hess[active_set] = min.(max.(hess[active_set], control.lower), control.upper)

        #Update fixed effect parameters that are in active_set
        for j in active_set

            # we also pass XG and y instead of XGgrp and ygrp for reasons of efficiency--see definition of special_quad
            cut = special_quad(res_iter, XG[:,j], βiter[j], j, grp, invVgrp, XGgrp)

            if hess[j] == hess_untrunc[j] #Outcome of Armijo rule can be computed analytically
                if j in 1:q
                    βiter[j] = cut / hess[j]
                elseif penalty == "lasso"
                    βiter[j] = soft_thresh(cut, λwtd[j]) / hess[j]
                else #Scad penalty
                    βiter[j] = scad_solution(cut, hess[j], λwtd[j], scada)
                end
            else #Must actually perform Armijo line search 
                num_arm += 1
                fct_iter, arm_con = armijo!(
                    XGgrp,
                    ygrp,
                    invVgrp,
                    βiter,
                    j,
                    q,
                    cut,
                    hess_untrunc[j],
                    hess[j],
                    penalty,
                    λwtd,
                    scada,
                    fct_iter,
                    arm_con,
                    control,
                )
            end
            # update residuals
            res_iter .-= (βiter[j] - βold[j])*XG[:, j] 
        end
        
        #---Optimization with respect to random effect parameters ----------------------------
        #------------------------------------------------------------------------------------
        if ψstr == "ident"
            #println(counter)
            #println(Liter)
            #println(σ²iter)
            Liter = L_scalar_update(
                XGgrp,
                ygrp,
                Zgrp,
                βiter,
                σ²iter,
                control
            )
        elseif ψstr == "diag"
            for s = 1:m
                L_update!(
                    Liter,
                    XGgrp,
                    ygrp,
                    Zgrp,
                    βiter,
                    σ²iter,
                    s,
                    control
                )
            end
        else  #ψstr == "sym"
            for i = 1:m
                for j = 1:i
                    #println("Updating ($i, $j) coordinate of L")
                    L_update!(
                        Liter,
                        XGgrp,
                        ygrp,
                        Zgrp,
                        βiter,
                        σ²iter,
                        (i, j),
                        control
                    )
                end
            end
        end

        #Optimization of σ²
        σ²iter = σ²update(XGgrp, ygrp, Zgrp, βiter, Liter, control)

        #Vector of variance/covariance parameters
        Lvec_iter = ndims(Liter) == 2 ? vec(Liter) : Liter
        varp_iter = vcat(Lvec_iter, sqrt(σ²iter))

        #Calculate new inverse variances 
        invV!(invVgrp, Zgrp, Liter, σ²iter)

        #Calculate new objective function
        neglike_iter = get_negll(invVgrp, ygrp, XGgrp, βiter)
        fct_iter = get_cost(neglike_iter, βiter[(q+1):end], penalty, λwtd[(q+1):end], scada)
        control.trace && println("After $counter cycles, objective is: $fct_iter")

        # Check convergence
        convβ = norm(βiter - βold) / (1 + norm(βiter))
        conv_varp = norm(varp_iter - varp_old) / (1 + norm(varp_iter))
        if penalty == "scad"
            conv_fct = 0  # Do not check function convergence for SCAD
        else
            conv_fct = abs(fct_iter - fct_old) / (1 + abs(fct_iter))
        end

        # If convergence criterion satisfied, then next iteration, update all fixed effects
        # If parameters still don't change after that, then the while condition won't get satisfied 
        # because do_all will be true
        if max(convβ, conv_varp, conv_fct) <= control.tol
            counter_in = 0
        end

    end

    if stopped == true
        @warn "Exceeded maximum active set size. Choose larger λ."
    end

    if arm_con > 0
        @warn "Armijo rule failed $arm_con times"
    end

    if counter == control.max_iter
        @warn "$counter iterations reached without convergence"
    end

    #In case of identity or diagonal covariance structure, form matrix version of L for future calculations
    if isa(Liter, Number)
        Lmat = Liter * I(m)
    elseif isa(Liter, Vector)
        Lmat = Diagonal(Liter)
    else #Symmetric, aka matrix 
        Lmat = Liter
    end

    #Get parameters and design matrices on the original scale 
    if standardize
        βstart[(q+1):end] = βstart[(q+1):end] ./ sdsG'
        βstart[1] = βstart[1] - sum(meansG' .* βstart[(q+1):end])
        βiter[(q+1):end] = βiter[(q+1):end] ./ sdsG'
        βiter[1] = βiter[1] - sum(meansG' .* βiter[(q+1):end])
        G = G .* sdsG .+ meansG 
        XG = [X G]
        XGgrp = Matrix[]
        for group in unique(grp)
            XGᵢ = XG[grp.==group, :]
            push!(XGgrp, XGᵢ)
        end
    end

    #Predicted random effects
    resid_fixed = [yᵢ - XGᵢ * βiter for (yᵢ, XGᵢ) in zip(ygrp, XGgrp)]
    u =
        sqrt(σ²iter) * [
            inv(Lmat' * Zᵢ' * Zᵢ * Lmat + σ²iter * I(m)) * Lmat' * Zᵢ' * residᵢ for
            (Zᵢ, residᵢ) in zip(Zgrp, resid_fixed)
        ]
    b = [Lmat * uᵢ for uᵢ in u] / sqrt(σ²iter)

    #Fitted values and residuals
    fittedgrp = [XGᵢ * βiter + Zᵢ * bᵢ for (XGᵢ, Zᵢ, bᵢ) in zip(XGgrp, Zgrp, b)]
    fitted = Vector{Float64}(undef, N)
    for (i, group) in enumerate(unique(grp))
        fitted[grp.==group] = fittedgrp[i]
    end
    resid = y - fitted

    #Number of covariance parameters
    nz_covpar = sum(Liter .!= 0)
    if isa(Liter, Matrix)
        n_covpar = m * (m + 1) / 2
    elseif isa(Liter, Vector)
        n_covpar = m
    else #scalar
        n_covpar = 1
    end
    n_covpar == nz_covpar ||
        println("$(n_covpar-nz_covpar) redundant variance/covariance parameters (set to 0)")

    #Number of non-zeros in final fixed effect estimates
    nz = sum(βiter .!= 0)

    #Criteria
    npar = sum(βiter .!= 0) + nz_covpar
    deviance = 2 * neglike_iter
    aic = deviance + 2 * npar
    bic = deviance + log(N) * npar

    #Return
    out = HDMModel(
        (X = X, G = G, Z = Z, y = y, grp = grp),
        wts,
        (βstart = βstart, Lstart = Lstart, σ²start = σ²start),
        -neglike_start,
        fct_start,
        nz_start,
        penalty,
        standardize,
        λ,
        scada,
        σ²iter,
        Lmat,
        βiter,
        b,
        fitted,
        resid,
        -neglike_iter,
        fct_iter,
        npar,
        nz,
        deviance,
        num_arm,
        arm_con,
        aic,
        bic,
        counter,
        ψstr,
        Lmat * Lmat',
        control,
    )

    return out

end

end
