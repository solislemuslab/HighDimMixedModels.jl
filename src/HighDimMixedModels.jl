module HighDimMixedModels

using LinearAlgebra
using Random
using Lasso
using Optim
using InvertedIndices #Allows negative indexing, like in R

export cov_start
export L_ident_update
export L_diag_update!

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
Finds an initial value for the covariance parameters 

ARGUMENTS
- Xgrp :: Vector of fixed effect design matrices for each group
- ygrp :: Vector of vector of responses for each group
- Zgrp :: Vector of random effects design matrix for each group
- β :: Initial iterate for fixed effect parameter vector (computed with Lasso ignoring group structure)

OUTPUT
- Assuming β is true fixed effect vector, estimate of L, and estimate of σ²

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
function L_diag_update!(L, Xgrp, ygrp, Zgrp, β, σ², s, var_int, thres)
    
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
Fits penalized linear mixed effect model 

ARGUMENTS
- G :: High dimensional design matrix for penalized fixed effects (not assumed to include column of ones)
- X :: Low dimensional design matrix for unpenalized fixed effects (not assumed to include column of ones)
- Z :: Design matrix for random effects
- grp :: Variable defining groups (i.e. factor)
- Ψstr :: One of :ident, :diag, :sym, specifying covariance structure of random effects
- λ :: Penalty parameter
- control :: control mutable struct with algorithm hyperparameters

OUTPUT
- Fitted model
"""
function lmmlasso(G, X, y, Z=X, grp, Ψstr, λ; control) 

    # --- Introductory checks --- 
    # ---------------------------------------------

    #Check that ψstr matches one of the options
    #Etc.


    
    # --- Intro allocations ---
    # ---------------------------------------------
    g = length(unique(grp)) #Number of groups
    N = length(y) #Total number of observations
    q = size(G, 2) #Number of penalized covariates
    p = size(X, 2) #Number of unpenalized covariates
    m = size(Z, 2) #Number of random effects
    Xaug = [ones(N) X] 
    XaugG = [Xaug G]
    #Grouped data
    Zgrp, XaugGgrp, ygrp = Matrix[], Matrix[], Vector[]
    for group in unique(grp)
        Zᵢ, XaugGᵢ, yᵢ = Z[grp .== group,:], XaugG[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(XaugGgrp, XaugGᵢ); push(ygrp, yᵢ)
    end
    

    # --- Initializing parameters ---
    # ---------------------------------------------

    #Initialized fixed effect parameters using Lasso that ignores random effects
    lassopath = fit(LassoPath, [X G], y; penalty_factor=[zeros(p); ones(q)])
    fpars = coef(lassopath; select=MinBIC()) #Fixed effects
    β = fpars[(p+2):end]

    #Initialize covariance parameters
    L, σ² = cov_start(XaugG, ygrp, Zgrp, fpars)


    # --- Calculate objective function for the starting values ---
    # ---------------------------------------------
    Vgrp = Vgrp(L, Zgrp, σ²)
    neglike = negloglike(Vgrp, ygrp, XaugGgrp, fpars)
    cost = neglike + λ*norm(β₀, 1)
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
    
    hess_diag = zeros(p+q+1)
    fpars_change = 
    

    #Algorithm parameters
    converged = 0
    counter_in = 0
    counter = 0

    while converged = 0 
        counter += 1
        println(counter,"...") 

        #Variables that are being updated
        fparsold = copy(fpars)
        βold = copy(β)
        costold = cost
        Lold = copy(L)
        σ²old = σ² 


        #---Optimization with respect to fixed effect parameters ----------------------------
        #------------------------------------------------------------------------------------

        #We'll only update fixed effect parameters in "active_set"--
        #see  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
        active_set = findall(fpars .== 0)
        
        if counter_in == 0 || counter_in > control.number 
            active_set = 1:(p+q+1)
            counter_in = 1
        else 
            counter_in += 1
        end

        XaugGactgrp = map(X -> X[:,active_set], XaugGgrp)
        hessian_diag!(Vgrp, XaugGactgrp, active_set, hess_diag)
        hess_diag_untrunc = copy(hess_diag)
        hess_diag[active_set] = max.(min.(hess_diag[active_set], control.lower), control.upper)

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
            control.trace > 3 && println(cost) 
        end
        β = fpars[(p+2):end] #Pull out new penalized coefficients

        control.trace > 3 && println("------------------------")


        #---Optimization with respect to covariance parameters ----------------------------
        #------------------------------------------------------------------------------------

        # calculations before the covariance optimization
        active_set = findall(fpars .== 0)
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
        Vgrp = V(L, Zgrp, σ²)
        neglike = negloglike(Vgrp, ygrp, XaugGgrp, fpars)
        cost = neglike + λ*norm(β, 1)

        #Compare to previous cost function...

    end
end




end
