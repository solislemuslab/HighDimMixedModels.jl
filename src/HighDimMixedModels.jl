module HighDimMixedModels

using LinearAlgebra
using Random
using Lasso #Needed to get initial estimates of the fixed effects
using Optim #Needed for univariate optimization in coordinated gradient descent algorithm
using InvertedIndices #Allows negative indexing, like in R
using Parameters #Supplies macro for structs with default field values

export cov_start
export L_ident_update
export L_diag_update!
export Control

"""
Returns covariance matrices of the responses, by group

ARGUMENTS
- L :: Parameters for random effect covariance matrix (can be scalar, vector, or lower triangular matrix)
- Zgrp :: Vector of random effects design matrix for each group
- σ² :: Variance of error

OUTPUT
- Vgrp :: List of length the number of groups, each of whose elements is the covariance matrix of the responses within a group
"""
function Vgrp(L, Zgrp, σ²)
    q = size(Zgrp[1])[2]

    if length(L) == 1
        Ψ = (L^2)I(q)
    elseif isa(L, Vector) && length(L) == q 
        Ψ = Diagonal(L.^2)
    else
        Ψ = L*L' 
    end
    
    Vgrp = Matrix[]
    for Zᵢ in Zgrp
        nᵢ = size(Zᵢ)[1]
        push!(Vgrp, Symmetric(Zᵢ * Ψ * Zᵢ' + σ² * I(nᵢ))) #Symmetric used to adjust for floating point error 
    end
    return Vgrp 
end


"""
Calculates residuals (list of residuals for each group) 
"""
function resid(Xgrp, ygrp, β, active_set = 1:length(β))
    Xactgrp = map(X -> X[:,active_set], Xgrp)
    residgrp = ygrp .- Xactgrp.*[β[active_set]]
    return residgrp
end

"""
Calculates x'M⁻¹x, where M is psd symmetric matrix
"""
quad_form_inv(M, x) = dot(x, cholesky(M)\x) 
#M will be one of the elements of Vgrp, so it should be pd
#We should probably implement a check for this, though 

"""
Calculates x'M⁻¹y, where M is psd symmetric matrix
"""
quad_form_inv2(M,x,y) = dot(x, cholesky(M)\y)
#M will be one of the elements of Vgrp, so it should be pd
#We should probably implement a check for this, though 

"""
Calculates the negative log-likelihod 
-l(ϕ̃) = -l(β, θ, σ²) = .5(Ntot*log(2π) + log|V| + (y-xβ)'V⁻¹(y-Xβ)) 

ARGUMENTS
- Vgrp :: Vector of length the number of groups, each of whose elements is the covariance matrix of the responses within a group
- ygrp :: Vector of vector of responses for each group
- X :: Vector of fixed effect design matrices for each group
- β :: Fixed effects

OUTPUT
- Value of the negative log-likelihood
"""
function negloglike(Vgrp, ygrp, Xgrp, β)

    detV = sum(map(x -> logabsdet(x)[1], Vgrp))
    residgrp = resid(Xgrp, ygrp, β)
    quadgrp = quad_form_inv.(Vgrp, residgrp)
    Ntot = sum(length.(ygrp))
    return .5(Ntot*log(2π) + detV + sum(quadgrp)) 

end

"""
Finds an initial value for the variance and covariance parameters 

ARGUMENTS
- Xgrp :: Vector of fixed effect design matrices for each group
- ygrp :: Vector of vector of responses for each group
- Zgrp :: Vector of random effects design matrix for each group
- β :: Initial iterate for fixed effect parameter vector (computed with Lasso ignoring group structure)

OUTPUT
- Assuming β is true fixed effect vector, MLE estimate of scalar L and scalar σ² as tuple (L, σ²)

"""
function cov_start(Xgrp, ygrp, Zgrp, β)
    
    # It can be shown that if we know η = sqrt(Ψ/σ²) and β, the MLE for σ² is 
    # given by the below function

    function σ²hat(Xgrp, ygrp, Zgrp, β, η) 
        Ṽgrp = Vgrp(η, Zgrp, 1)
        residgrp = resid(Xgrp, ygrp, β)
        quadgrp = quad_form_inv.(Ṽgrp, residgrp)
        Ntot = sum(length.(ygrp))
        σ² = sum(quadgrp)/Ntot
        return(σ²)
    end
    
    # We can now profile the likelihood and optimize with respect to η 

    function profile(η)
        σ² = σ²hat(Xgrp, ygrp, Zgrp, β, η)
        L = η*sqrt(σ²)
        negloglike(Vgrp(L, Zgrp, σ²), ygrp, Xgrp, β) 
    end

    result = optimize(profile, 0.0001, 1.0e4) #will need to fix
    
    if Optim.converged(result)
        η = Optim.minimizer(result)
    else
        error("Convergence for η not reached")
    end

    σ² = σ²hat(Xgrp, ygrp, Zgrp, β, η)
    L = η*sqrt(σ²)
    return L, σ²

end


"""
Calculates active_set entries of the diagonal of Hessian matrix for fixed effect parameters 
and updates `hess_diag` with these values--see 
"""
function hessian_diag!(Vgrp, Xactgrp, active_set, hess_diag)
    quadgrp = quad_form_inv.(Vgrp, Xactgrp) 
    hess_diag[active_set] = quadgrp
    return nothing 
end


"""
Soft Threshold
"""
soft_thresh(z,g) = sign(z)*max(abs(z)-g,0)


"""
Armijo Rule
"""
function armijo(Xgrp, ygrp, Vgrp, fpars, j, cut, hold, hnew, cost, p, converged)
    fparnew = copy(fpars)
   
    #Calculate dk
    grad = fpars[j]*hold - cut
    if j in 1:p+1
        dk = -grad/hnew
    else
        dk = median([(λ - grad)/hnew, -fpars[j], (-λ - grad)/hnew])
    end

    if dk!=0
        #Calculate Δk
        if j in 1:p+1
            Δk = dk*grad + control.γ*dk^2*hnew
        else
            Δk = dk*grad + control.γ*dk^2*hnew + λ*(abs(fpars[j]+dk)-abs(fpars[j]))
        end
        
        #Armijo line search
        for l in 0:control.max_armijo
            fparsnew[j] = fpars[j] + control.a_init*control.Δ^l*dk
            costnew = negloglike(Vgrp, ygrp, Xgrp, fparnew) + λ*norm(fparsnew[p+2:end], 1)
            addΔ = control.a_init*control.Δ^l*control.ρ*Δk
            if costnew <= cost + addΔ
                fpars[j] = fparsnew[j]
                cost = costnew
                break 
            end
            if l == control.max_armjio
                trace > 2 && println("Armijo for coordinate $(j) of fixed parameters not successful") 
                converged = converged+2 
            end
        end
    end
    
    return (fpars = fpars , cost = cost, converged = converged)
end

""" 
Update of L for identity covariance structure
""" 
function L_ident_update(Xgrp, ygrp, Zgrp, β, σ², var_int, thres)

    profile(L) = negloglike(Vgrp(L, Zgrp, σ²), ygrp, Xgrp, β)
    result = optimize(profile, var_int[1], var_int[2]) #Will need to fix
    
    Optim.converged(result) || error("Minimization with respect to $(s)th coordinate of L failed to converge")
    min = Optim.minimizer(result)

    if min < thres
        L = 0
        println("L was set to 0 (no group variation)")
    else
        L = min
    end

    return L
end


"""
Update of coordinate s of L for diagonal covariance structure

ARGUMENTS
- L :: A vector of parameters which will be updated by the function
- s :: The coordinate of L that is being updated (number between 1 and length(L))
"""
function L_diag_update!(L, Xgrp, ygrp, Zgrp, β, σ², s, var_int = (0, 1e6), thres=1e-4)
    
    Lcopy = copy(L)
    
    function profile(x)
        Lcopy[s] = x 
        negloglike(Vgrp(Lcopy, Zgrp, σ²), ygrp, Xgrp, β) 
    end
    result = optimize(profile, var_int[1], var_int[2]) #Will need to fix

    Optim.converged(result) || error("Minimization with respect to $(s)th coordinate of L failed to converge")
    min = Optim.minimizer(result)
    
    if min < thres
        L[s] = 0
        println("$(s) coordinate of L was set to 0")
    else
        L[s] = min
    end

    return Nothing

end

""" 
Update of L for general symmetric positive definite covariance structure
ARGUMENTS
- L :: A lower triangular matrix of parameters which will be updated by the function
- coords :: Tuple representing the coordinates of the entry of L that is being updated
"""
function L_sym_update!(L, Xgrp, ygrp, Zgrp, β, σ², coords, var_int, cov_int, thres)
    
    Lcopy = copy(L)
    int = coords[1]==cords[2] ? var_int : cov_int #Are we minimizing a covariance parameter or a variance parameters
    function profile(x)
        Lcopy[coords[1], coords[2]] = x 
        negloglike(Vgrp(Lcopy, Zgrp, σ²), ygrp, Xgrp, β) 
    end

    result = optimize(profile, int[1], int[2]) #Will need to fix

    Optim.converged(result) || error("Minimization with respect to $(coords) entry of L failed to converge")
    min = Optim.minimizer(result)
    
    if min < thres
        L[coords[1], coords[2]] = 0
        println("$(coords) entry of L was set to 0")
    else
        L[coords[1], coords[2]] = min
    end

    return Nothing
end

"""
Update of σ²
"""
function σ²update(Xgrp, ygrp, Zgrp, β, L, var_int)
    
    profile(σ²) = negloglike(Vgrp(L, Zgrp, σ²), ygrp, Xgrp, β)
    result = optimize(profile, var_int[1]^2, var_int[2]^2) #Will need to fix
    
    Optim.converged(result) || error("Minimization with respect to σ² failed to converge")

    return Optim.minimizer(result)

end

"""
Algorithm Hyper-parameters

- tol :: Convergence tolerance
- seed :: Random seed for cross validation for estimating initial fixed effect parameters using Lasso
- trace :: Integer. 1 prints no output, 2 prints warnings, and 3 prints the current iterate and function value and warnings
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
- G :: High dimensional design matrix for penalized fixed effects (not assumed to include column of ones) (REQUIRE)
- X :: Low dimensional design matrix for unpenalized fixed effects (assumed to include column of ones) (REQUIRED)
- y :: Vector of responses (REQUIRED)
- grp :: Vector of strings of same length as y assigning each observation to a particular group (REQUIRED)
- Z :: Design matrix for random effects (optional, default is all columns of X)
Keyword:
- λ :: Positive regularizing penalty (optional, default is 10.0)
- init_coef :: Tuple with three elements: starting value for fixed effects, random effect, and error variance parameters
- Ψstr :: One of :ident, :diag, :sym, specifying covariance structure of random effects. Default is :diag.
- Control :: Struct with fields for hyperparameters of the algorithm

OUTPUT
- Fitted model
"""
function lmmlasso(G::Matrix{Real}, X::Matrix{Real}, y::Vector{Real}, grp::Vector{String}, Z=X::Matrix{Real}; 
    λ::Float64=10.0, init_coef::Union{Vector, Nothing}=nothing, Ψstr::Symbol=:diag, Control=Control()) 

    # --- Introductory checks --- 
    # ---------------------------------------------
    N = length(y) #Total number of observations
    @assert size(G, 1) == N "G and y incompatable dimension"
    @assert size(X, 1) == N "X and y incompatable dimension"
    @assert size(Z, 1) == N "Z and y incompatable dimension"
    @assert length(grp) == N "grp and y incompatable dimension"

    @assert X[:,1] == ones(N) "First column of X must be all ones"
    groups = unique(grp)
    g = length(groups) #Number of groups
    @assert g > 1 "Only one group, no covariance parameters can be estimated"
    
    @assert λ > 0 "λ is regularization parameter, must be positive"

    #Check that ψstr matches one of the options
    #Etc.

    @assert ψstr in [:ident, :diag, :sym] "ψstr must be one of :ident, :diag, or :sym"
    Control::HighDimMixedModels.Control 
    @assert Control.optimize_method in [:Brent, :GoldenSection] "Control.optimize_method must be one of :Brent or :GoldenSection"
    if init_coef !== nothing
        @assert length(init_coef) == 3 "init_coef must be of length 3"
        @assert length(init_coef)
    end
    

    # --- Intro allocations -----------------------
    # ---------------------------------------------
    q = size(G, 2) #Number of penalized covariates
    p = size(X, 2) - 1 #Number of unpenalized covariates (does not include intercept)
    m = size(Z, 2) #Number of covariates associated with random effects
    XG = [X G]
    
    #Grouped data
    Zgrp, XGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XGᵢ, yᵢ = Z[grp .== group,:], XG[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(XGgrp, XGᵢ); push(ygrp, yᵢ)
    end
    
    # --- Initializing parameters ---
    # ---------------------------------------------
    if y === nothing
        #Initialize fixed effect parameters using Lasso that ignores random effects
        Random.seed!(Control.seed)
        lassopath = fit(LassoPath, [X[:,2:end] G], y; penalty_factor=[zeros(p); ones(q)])
        fpars = coef(lassopath; select=MinCVmse(lassopath, 10)) #Fixed effects

        #Initialize covariance parameters
        L, σ² = cov_start(XGgrp, ygrp, Zgrp, fpars)
    else
        fpars, L, σ² = init_coef
    end
    β = fpars[(p+2):end]


    # --- Calculate objective function for the starting values ---
    # ---------------------------------------------
    Vgrp = Vgrp(L, Zgrp, σ²)
    neglike = negloglike(Vgrp, ygrp, XaugGgrp, fpars)
    cost = neglike + λ*norm(β, 1)
    println("Cost at initialization: $(cost)")


    # --- Coordinate Gradient Descent -------------
    # ---------------------------------------------

    #Some allocations 
    if Ψstr == :sym
        L = LowerTriangular(L*I(m))
    elseif Ψstr == :diag
        L = fill(L, m) 
    elseif Ψstr != :ident
        error("ψstr must be one of :sym, :diag, or :ident")
    end
    

    #Algorithm allocations
    hess_diag = zeros(p+q+1)
    converged = false
    counter_in = 0
    counter = 0

    while converged == false 
        counter += 1
        Control.trace == 2 && println("Outer iteration $counter") 

        #Variables that are being updated
        fparsold = copy(fpars)
        Lold = copy(L)
        σ²old = σ² 
        costold = cost


        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"--
        #see  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(fpars .== 0)
        
        if counter_in == 0 || counter_in > Control.act_min 
            active_set = 1:(p+q+1)
            counter_in = 1
        else 
            counter_in += 1
        end

        XaugGactgrp = map(X -> X[:,active_set], XaugGgrp)
        hessian_diag!(Vgrp, XaugGactgrp, active_set, hess_diag)
        hess_diag_untrunc = copy(hess_diag)
        hess_diag[active_set] = max.(min.(hess_diag[active_set], Control.lower), Control.upper)

        #Update fixed effect parameters that are in active_set
        for j in active_set 

            XaugGgrpⱼ = map(X -> X[:j], XaugGgrp)
            XaugGgrp₋ⱼ = map(X -> X[:,Not(j)], XaugGgrp)
            rgrp₋ⱼ = resid(XaugGgrp₋ⱼ, ygrp, fpars[Not(j)])
            cut = sum(quad_form_inv2(Vgrp, rgrp₋ⱼ, XaugGgrpⱼ)) 

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
            Control.trace > 3 && println(cost) 
        end
        β = fpars[(p+2):end] #Pull out new penalized coefficients

        Control.trace > 3 && println("------------------------")


        #---Optimization with respect to covariance parameters ----------------------------
        #------------------------------------------------------------------------------------

        # calculations before the covariance optimization
        active_set = findall(fpars .== 0)
        resgrp = resid(XaugGgrp, ygrp, fpars, active_set)
        
        # Optimization of L
        if ψstr == :ident
            L = L_ident_update(XaugGgrp, ygrp, Zgrp, fpars, σ², Control.var_int, Control.thres)
        elseif ψstr == :diag
            foreach( s -> L_diag_update!(L, XaugGgrp, ygrp, Zgrp, fpars, σ², s, Control.var_int, Control.thres), 1:m)
        else  #ψstr == :sym
            tuples = collect(Iterators.product(1:m, 1:m)) #Matrix of tuples
            tuples = tuples[tril!(trues((m,m)), 0)] #Vector of tuples (i, j), where i≥j
            foreach( coords -> L_sym_update!(L, XaugGgrp, ygrp, Zgrp, fpars, σ², coords, Control.var_int, Control.cov_int, Control.thres), 
                    tuples)
        end

        Control.trace > 3 && println("------------------------")

        # Optimization of σ²
        σ² = σ²update(XaugGgrp, ygrp, Zgrp, fpars, L, Control.var_int)
        Control.trace > 3 && println("------------------------")

        #Calculate new cost function
        Vgrp = V(L, Zgrp, σ²)
        neglike = negloglike(Vgrp, ygrp, XaugGgrp, fpars)
        cost = neglike + λ*norm(β, 1)

        #Compare to previous cost function...

    end
end




end
