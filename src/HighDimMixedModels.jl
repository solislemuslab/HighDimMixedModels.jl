module HighDimMixedModels

using Statistics
using LinearAlgebra
using Random
using InvertedIndices 
using Optim: Optim, optimize
using Parameters: @with_kw
using MLBase: Kfold 
using Lasso: Lasso
using StatsAPI
export 
    Control, 
    HDMModel, 
    hdmm, 
    coef, 
    coeftable, 
    deviance, 
    loglikelihood, 
    nobs, 
    residuals, 
    fitted, 
    aic, 
    bic

include("helpers.jl")
include("structs.jl")
include("statsbase.jl")

"""
Fits penalized linear mixed effect model 

ARGUMENTS
Positional: 
- X :: Low dimensional design matrix for unpenalized fixed effects (assumed to include column of ones) (REQUIRED)
- G :: High dimensional design matrix for penalized fixed effects (assumed to not include column of ones) (REQUIRE)
- y :: Vector of responses (REQUIRED)
- grp :: Vector of strings of same length as y assigning each observation to a particular group (REQUIRED)
- Z :: Design matrix for random effects (default is all columns of X)
NOTE: Z is not expected to be given in block diagonal form. It should be a vertical stack of subject design matrices Z₁, Z₂, ...

Keyword:
- standardize :: boolean (default true), whether to standardize design matrices before performing algorithm. 
- penalty :: One of "scad" (default) or "lasso"
- λ :: Positive regularizing penalty (default is 10.0)
- scada :: Extra tuning parameter for the SCAD penalty (default is 3.7, ignored if penalty is "lasso"
- wts :: Vector of length number of penalized coefficients. Strength of penalty on covariate j is λ/wⱼ (Default is vector of 1's)
- init_coef :: Named tuple of form (β, L, σ²) giving initial values for parameters. If unspecified, then inital values for parameters are 
calculated as follows: first, cross-validated LASSO that ignores grouping structure is performed to obtain initial estimates of the 
fixed effect parameters. Then, the random effect parameters are initialized as MLEs assuming the LASSO estimates are true fixed effect parameters.
- ψstr :: One of "diag" (default), "ident", or "sym", specifying covariance structure of random effects 
- control :: Struct with fields for hyperparameters of the algorithm 

OUTPUT
- Fitted model
"""
function hdmm(X::Matrix{<:Real}, G::Matrix{<:Real}, y::Vector{<:Real}, 
    grp::Vector{<:Union{String, Int64}}, Z::Matrix{<:Real}=X;
    standardize=true, penalty::String="scad", λ::Real=10.0, scada::Real=3.7, 
    wts::Union{Vector, Nothing}=nothing, init_coef::Union{Vector,Nothing}=nothing, 
    ψstr::String="diag", control::Control=Control())

    ##Get dimensions
    N = length(y) # Total number of observations
    p = size(G, 2) # Number of penalized covariates 
    q = size(X, 2) # Number of unpenalized covariates 
    m = size(Z, 2) # Number of covariates associated with random effects (less than or equal to q) 
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

    #Intercept included check
    @assert X[:, 1] == ones(N) "First column of X must be all ones"
    Z_int = (Z[:, 1] == ones(N)) #Bool for whether Z includes intercept (i.e. whether there's a random intercept)

    #Check whether columns of Z are a subset of the columns of X, or is this not necessary?

    #Penalty related checks
    @assert penalty in ["scad", "lasso"] "penalty must be one of \"scad\" or \"lasso\""
    @assert λ > 0 "λ is regularization parameter, must be positive"
    @assert scada > 0 "scada is regularization parameter, must be positive"
    wts = (wts === nothing ? fill(1, p) : wts)
    @assert all(wts .> 0) "Weights must be positive"
    @assert length(wts) == p "Number of weights must equal number of penalized covariates"
    # Create the vector of penalty parameters of length q + p, i.e. one for each fixed effect
    λwtd = [zeros(q); λ ./ wts]
    @assert ψstr in ["ident", "diag", "sym"] "ψstr must be one of \"ident\", \"diag\", or \"sym\""
    @assert control.optimize_method in [:Brent, :GoldenSection] "Control.optimize_method must be one of :Brent or :GoldenSection"

    #Check that initial coefficients, if provided, make sense
    if init_coef !== nothing
        @assert length(init_coef) == 3 "init_coef must be of length 3"
        @assert init_coef.β isa Vector{Real} "init_coef.β must be a vector of reals"
        @assert length(init_coef.β) == q + p "init_coef.β must be of length q + p"
        @assert init_coef.L isa Real "init_coef.L must be a real number"
        @assert init_coef.L > 0 "init_coef.L must be positive"
        @assert init_coef.σ² isa Real "init_coef.σ² must be a real number"
        @assert init_coef.σ² > 0 "init_coef.σ² must be positive"
    end

    # --- Introductory allocations ----------------
    # ---------------------------------------------

    #Merge unpenalized and penalized design matrices
    XG = [X G]

    #Standardize design matrices if so desired
    if standardize
        XGor = copy(XG)
        meansx = mean(XG[:, Not(1)], dims=1)
        sdsx = std(XG[:, Not(1)], dims=1)
        XG = (XG[:, Not(1)] .- meansx) ./ sdsx
        XG = [fill(1, N) XG]

        Zor = copy(Z)
        if Z_int 
            meansz = mean(Z[:, Not(1)], dims=1)
            sdsz = std(Z[:, Not(1)], dims=1)
            Z = (Z[:, Not(1)] .- meansz) ./ sdsz
            Z = [fill(1, N) Z]
        else 
            meansz = mean(Z, dims=1)
            sdsz = std(Z, dims=1)
            Z = (Z .- meansz) ./ sdsz
        end
    end

    #Get grouped data, i.e. lists of matrices/vectors
    Zgrp, XGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp.==group, :], XG[grp.==group, :], y[grp.==group]
        push!(Zgrp, Zᵢ)
        push!(XGgrp, XGᵢ)
        push!(ygrp, yᵢ)
    end

    # --- Initialize parameters -------------------
    # ---------------------------------------------
    if init_coef === nothing
        #Initialize fixed effect parameters using standard, cross-validated Lasso which ignores random effects
        lassopath = Lasso.fit(Lasso.LassoModel, XG[:, Not(1)], y; 
                                maxncoef = max(2*N, 2*p), #see https://github.com/JuliaStats/Lasso.jl/issues/54
                                penalty_factor=[zeros(q - 1); 1 ./ wts], 
                                select=Lasso.MinCVmse(Kfold(N, 10)))
        βstart = Lasso.coef(lassopath) #Fixed effects
        #Initialize covariance parameters
        Lstart, σ²start = cov_start(XGgrp, ygrp, Zgrp, βstart)
    else
        βstart, Lstart, σ²start = init_coef
    end

    #Number of non-zeros in initial fixed-effect estimates 
    nz_start = sum(βstart .!= 0)

    #Get L in form specified by ψstr
    if ψstr == "sym"
        Lstart = Matrix(Lstart * I(m))
    elseif ψstr == "diag"
        Lstart = fill(Lstart, m)
    elseif ψstr != "ident"
        error("ψstr must be one of \"sym\", \"diag\", or \"ident\"")
    end

    # --- Calculate cost for the starting values ---
    # ---------------------------------------------
    invVgrp = Vector{Matrix}(undef, g)
    invV!(invVgrp, Zgrp, Lstart, σ²start)
    neglike_start = get_negll(invVgrp, ygrp, XGgrp, βstart)
    fct_start = get_cost(neglike_start, βstart[(q+1):end], penalty, λwtd[(q+1):end], scada)
    control.trace > 2 && println("Cost at initialization: $fct_start")

    # --- Coordinate Gradient Descent -------------
    # ---------------------------------------------

    #Algorithm allocations
    βiter = copy(βstart)
    Liter = copy(Lstart)
    σ²iter = σ²start
    Lvec_iter = ndims(Liter) == 2 ? vec(Liter) : Liter #L as vector if ψstr == "sym"
    varp_iter = vcat(Lvec_iter, sqrt(σ²iter)) #Vector of covariance parameters
    neglike_iter = neglike_start
    fct_iter = fct_start
    hess = zeros(q+p) #Container for future calculations
    mat = zeros(g, q+p) #Container for future calculation

    convβ = norm(βiter)^2
    conv_varp = norm(varp_iter)^2
    conv_fct = fct_iter

    stopped = false #Make true if we start interpolating the data
    do_all = false
    counter_in = 0
    counter = 0
    num_arm = 0 #Goes up by 1 every time Armijo needs to be performed (cannot be computed analytically)
    arm_con = 0 #Goes up by 1 every time an Armijo step fails to converge

    while (max(convβ, conv_varp, conv_fct) > control.tol || !do_all) && (counter < control.max_iter)

        counter += 1

        #Keep copy of variables that are being updated for convergence checking
        βold = copy(βiter)
        varp_old = copy(varp_iter)
        fct_old = fct_iter

        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"
        #See  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(βiter .!= 0)

        # If the active set is larger than the total sample size and we're a few iterations in,
        # that means we're converging towards a solution that interpolates the data, which is bad
        if length(active_set) > N && λ > 0 && counter > 2
            stopped = true
            break
        end
        
        # We make the active set include all covariates every act_num iterations
        if counter_in == 0 || counter_in > control.act_num
            active_set = 1:(p+q)
            counter_in = 1
            do_all = true
        else
            do_all = false
            counter_in += 1
        end

        #Calculate the Hessian for the coordinates of the active set
        hessian_diag!(XGgrp, invVgrp, active_set, hess, mat[:,active_set])
        hess_untrunc = copy(hess)
        hess[active_set] = min.(max.(hess[active_set], control.lower), control.upper)

        #Update fixed effect parameters that are in active_set
        for j in active_set
            # we also pass XG and y instead of XGgrp and ygrp for reasons of efficiency--see definition of special_quad
            cut = special_quad(XG, y, βiter, j, invVgrp, XGgrp, grp) 

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
                fct_iter, arm_con = armijo!(XGgrp, ygrp, invVgrp, βiter, j, q, cut, hess_untrunc[j],
                    hess[j], penalty, λwtd, scada, fct_iter, arm_con, control)
            end
        end

        #Calculate new objective function and print if trace > 2
        neglike_iter = get_negll(invVgrp, ygrp, XGgrp, βiter)
        fct_iter = get_cost(neglike_iter, βiter[(q+1):end], penalty, λwtd[(q+1):end], scada)
        control.trace > 2 && println("After updating fixed effects, cost is $fct_iter")

        #---Optimization with respect to random effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        if ψstr == "ident"
            #println(counter)
            #println(Liter)
            #println(σ²iter)
            Liter = L_ident_update(XGgrp, ygrp, Zgrp, βiter, σ²iter,
                control.var_int, control.thres)
        elseif ψstr == "diag"
            for s in 1:m
                L_diag_update!(Liter, XGgrp, ygrp, Zgrp, βiter, σ²iter, s,
                    control.var_int, control.thres)
            end
        else  #ψstr == "sym"
            for i in 1:m
                for j in 1:i
                    #println("Updating ($i, $j) coordinate of L")
                    L_sym_update!(Liter, XGgrp, ygrp, Zgrp, βiter, σ²iter, (i, j),
                        control.var_int, control.cov_int, control.thres)
                end
            end
        end

        #Optimization of σ²
        σ²iter = σ²update(XGgrp, ygrp, Zgrp, βiter, Liter, control.var_int)

        #Vector of variance/covariance parameters
        Lvec_iter = ndims(Liter) == 2 ? vec(Liter) : Liter
        varp_iter = vcat(Lvec_iter, sqrt(σ²iter))

        #Calculate new inverse variances 
        invV!(invVgrp, Zgrp, Liter, σ²iter)

        #Calculate new objective function
        neglike_iter = get_negll(invVgrp, ygrp, XGgrp, βiter)
        fct_iter = get_cost(neglike_iter, βiter[(q+1):end], penalty, λwtd[(q+1):end], scada)

        #Inserted to prevent covergence issues
        # if neglike_iter < 0
        #     stopped = true
        #     break
        # end

        #Check convergence
        convβ = norm(βiter - βold) / (1 + norm(βiter))
        conv_varp = norm(varp_iter - varp_old) / (1 + norm(varp_iter))
        conv_fct = abs(fct_iter - fct_old) / (1 + abs(fct_iter))

        #If convergence criterion satisfied, then next iteration, update all fixed effects
        #If parameters still doesn't change after that, then the while condition won't get satisfied 
        #because do_all will be true
        if max(convβ, conv_varp, conv_fct) <= control.tol
            counter_in = 0 
        end

    end
    
    if stopped == true
        error("More active fixed effects than samples. Choose larger λ.")
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
        βiter[Not(1)] = βiter[Not(1)] ./ sdsx'
        βiter[1] = βiter[1] - sum(meansx' .* βiter[Not(1)])
        if Z_int
            Lmat = Diagonal([1; vec(1 ./ sdsz)]) * Lmat
        else
            Lmat = Diagonal(vec(1 ./ sdsz)) * Lmat
        end

        XG = XGor
        Z = Zor
        Zgrp, XGgrp = Matrix[], Matrix[]
        for group in unique(grp)
            Zᵢ, XGᵢ = Z[grp.==group, :], XG[grp.==group, :]
            push!(Zgrp, Zᵢ)
            push!(XGgrp, XGᵢ)
        end
    end

    #Predicted random effects
    resid_fixed = [yᵢ - XGᵢ * βiter for (yᵢ, XGᵢ) in zip(ygrp, XGgrp)]
    u = sqrt(σ²iter) * [inv(Lmat' * Zᵢ' * Zᵢ * Lmat + σ²iter * I(m)) * Lmat' * Zᵢ' * residᵢ for (Zᵢ, residᵢ) in zip(Zgrp, resid_fixed)]
    b = [Lmat * uᵢ for uᵢ in u] / sqrt(σ²iter)

    #Fitted values and residuals
    fittedgrp = [XGᵢ * βiter + Zᵢ * bᵢ for (XGᵢ, Zᵢ, bᵢ) in zip(XGgrp, Zgrp, b)]
    fitted = Vector{Float64}(undef, N)
    for (i, group) in enumerate(unique(grp))
        fitted[grp.==group] = fittedgrp[i]
    end
    resid = y-fitted

    #Number of covariance parameters
    nz_covpar = sum(Liter .!= 0)
    if isa(Liter, Matrix)
        n_covpar = m * (m + 1) / 2
    elseif isa(Liter, Vector)
        n_covpar = m
    else #scalar
        n_covpar = 1
    end
    n_covpar == nz_covpar || println("$(n_covpar-nz_covpar) redundant variance/covariance parameters (set to 0)")

    #Number of non-zeros in final fixed effect estimates
    nz = sum(βiter .!= 0)

    #Criteria
    npar = sum(βiter .!= 0) + nz_covpar
    deviance = 2 * neglike_iter
    aic = deviance + 2 * npar
    bic = deviance + log(N) * npar

    #Return
    out = HDMModel((X=X, G=G, Z=Z, y=y, grp=grp), wts, (βstart=βstart, Lstart=Lstart, σ²start=σ²start), 
                -neglike_start, fct_start, nz_start, penalty,  λ, scada, σ²iter, Lmat, βiter, b, fitted,
                resid, -neglike_iter, fct_iter, npar, nz, deviance,  arm_con,
                num_arm, aic,  bic, counter,  ψstr, Lmat * Lmat', control)

    return out

end

end
