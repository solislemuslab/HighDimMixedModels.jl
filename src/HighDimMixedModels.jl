module HighDimMixedModels

using LinearAlgebra
using Random
using Lasso
using Optim
using InvertedIndices #Allows negative indexing, like in R

export cov_start
export L_ident_update

"""
Covariance matrices of the responses

ARGUMENTS
- L :: Parameters for random effect covariance matrix (can be scalar, vector, or lower triangular matrix)
- Zgrp :: Vector of random effects design matrix for each group
- σ² :: Variance of error

OUTPUT
- invVgrp :: List of length the number of groups, each of whose elements is the inverse covariance matrix of the responses within a group
"""
function invV(L, Zgrp, σ²)
    q = size(Zgrp[1])[2]

    if length(L) == 1
        Ψ = L[1]^2
    elseif isa(L, Vector) && length(L) == q 
        Ψ = Diagonal(L.^2)
    else
        ϕ = L*L' 
    end
    
    invVgrp = Matrix[]
    for Zᵢ in Zgrp
        nᵢ = size(Zᵢ)[1]
        push!(invVgrp, inv(Zᵢ * Ψ * Zᵢ' + σ² * I(nᵢ))) #Should we check singularity? Not sure
    end
    return invVgrp 
end

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
function negloglike(invVgrp, ygrp, Xgrp, β)

    det_invV = sum(map(x -> logabsdet(x)[1], invVgrp))
    quads = @. transpose(ygrp - Xgrp * [β]) * invVgrp * (ygrp - Xgrp * [β])
    Ntot = sum(length.(ygrp))
    return .5(Ntot*log(2π) - det_invV + sum(quads)) 

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
        invṼgrp = invV(η, Zgrp, 1)
        quads = @. transpose(ygrp - Xgrp*[β]) * invṼgrp * (ygrp - Xgrp * [β])
        Ntot = sum(length.(ygrp))
        σ² = sum(quads)/Ntot
        return(σ²)
    end
    
    # We can now profile the likelihood and optimize with respect to η 

    function profile(η)
        σ² = σ²hat(Xgrp, ygrp, Zgrp, β, η[1])
        L = η[1]*sqrt(σ²)
        negloglike(invV(L, Zgrp, σ²), ygrp, Xgrp, β) 
    end

    result = optimize(profile, [0,], [Inf,], [1.,], Fminbox())
    
    if Optim.converged(result)
        η = Optim.minimizer(result)[1]
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
function hessian_diag!(invVgrp, Xactgrp, active_set, hess_diag)
    diag_quads = @. diag(transpose(Xactgrp) * invVgrp * Xactgrp)
    hess_diag[active_set] = sum(diag_quads)
    return nothing 
end


soft_thresh(z,g) = sign(z)*max(abs(z)-g,0)


"""
Calculates residuals 
"""
function resid(Xgrp, ygrp, β, active_set)
    Xactgrp = map(X -> X[:,active_set], Xgrp)
    residgrp = ygrp .- Xactgrp.*[β[active_set]]
    return residgrp
end



function armijo(Xgrp, ygrp, invVgrp, fpars, j, cut, hold, hnew, cost, p, converged)
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
            costnew = negloglike(invVgrp, ygrp, Xgrp, fparnew) + λ*norm(fparsnew[p+2:end], 1)
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


function L_ident_update(Xgrp, ygrp, Zgrp, β, σ²)

    profile(L) = negloglike(invV(L, Zgrp, σ²), ygrp, Xgrp, β)
    result = optimize(profile, [0,], [Inf,], [1.,], Fminbox())
    
    if Optim.converged(result)
        L = Optim.minimizer(result)[1]
    else
        error("Convergence for L not reached")
    end

    return L
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
#function lmmlasso(G, X, y, Z=X, grp, Ψstr, λ; control) 

#     # --- Introductory checks --- 
#     # ---------------------------------------------

#     #Check that ψstr matches one of the options
#     #Etc.


    
#     # --- Intro allocations ---
#     # ---------------------------------------------
#     g = length(unique(grp)) #Number of groups
#     N = length(y) #Total number of observations
#     q = size(G, 2) #Number of penalized covariates
#     p = size(X, 2) #Number of unpenalized covariates
#     m = size(Z, 2) #Number of random effects
#     Xaug = [ones(N) X] 
#     XaugG = [Xaug G]
#     #Grouped data
#     Zgrp, XaugGgrp, ygrp = Matrix[], Matrix[], Vector[]
#     for group in unique(grp)
#         Zᵢ, XaugGᵢ, yᵢ = Z[grp .== group,:], XaugG[grp .== group,:], y[grp .== group]
#         push!(Zgrp, Zᵢ); push!(XaugGgrp, XaugGᵢ); push(ygrp, yᵢ)
#     end
    

#     # --- Initializing parameters ---
#     # ---------------------------------------------

#     #Initialized fixed effect parameters using Lasso that ignores random effects
#     lassopath = fit(LassoPath, [X G], y; penalty_factor=[zeros(p); ones(q)])
#     fpars = coef(lassopath; select=MinBIC()) #Fixed effects
#     β₀ = fpars[(p+2):end]

#     #Initialize covariance parameters
#     L, σ² = cov_start(XaugG, ygrp, Zgrp, fpars)


#     # --- Calculate objective function for the starting values ---
#     # ---------------------------------------------
#     invVgrp = invV(L, Zgrp, σ²)
#     neglike = negloglike(invVgrp, ygrp, XaugGgrp, fpars)
#     cost = neglike + λ*norm(β₀, 1)
#     println("Cost at initialization: $(cost)")


#     # --- Coordinate Gradient Descent -------------
#     # ---------------------------------------------

#     #Some allocations 
#     if Ψstr == :sym
#         L = LowerTriangular(L*I(m))
#     elseif Ψstr == :diag
#         L = fill(L, m) 
#     elseif Ψstr != :ident
#         error("ψstr must be one of :sym, :diag, or :ident")
#     end
    
#     hess_diag = zeros(p+q+1)
#     fpars_change = 
    

#     #Algorithm parameters
#     converged = 0
#     counter_in = 0
#     counter = 0

#     while 
#         counter += 1
#         println(counter,"...") 

#         #Variables that are being updated
#         fparsold = copy(fpars)
#         costold = cost
#         Lold = copy(L)
#         σ²old = σ² 


#         #---Optimization with respect to fixed effect parameters ----------------------------
#         #------------------------------------------------------------------------------------

#         #We'll only update fixed effect parameters in "active_set"--
#         #see  page 53 of lmmlasso dissertation and Meier et al. (2008) and Friedman et al. (2010).
#         active_set = findall(fpars .== 0)
        
#         if counter_in == 0 || counter_in > control.number 
#             active_set = 1:(p+q+1)
#             counter_in = 1
#         else 
#             counter_in += 1
#         end

#         XaugGactgrp = map(X -> X[:,active_set], XaugGgrp)
#         hessian_diag!(invVgrp, XaugGactgrp, active_set, hess_diag)
#         hess_diag_untrunc = copy(hess_diag)
#         hess_diag[active_set] = max.(min.(hess_diag[active_set], control.lower), control.upper)

#         invVXaugGactgrp = invVgrp .* XaugGactgrp

#         #Update fixed effect parameters that are in active_set
#         for j in active_set 

#             XaugGgrp₋ⱼ = map(X -> X[:,Not(j)], XaugGgrp)
#             rgrp = ygrp .- XaugGgrp₋ⱼ * fpars[Not(j)]
#             cut = sum( transpose.(rgrp) .* map(X -> X[:, j], invVXaugGactgrp) ) 

#             if hess_diag[j] == hess_diag_untrunc[j] #Outcome of Armijo rule can be computed analytically
#                 if j in 1:p+1
#                     fpar[j] = cut/hess_diag[j]
#                 else
#                     fpar[j] = soft_thresh(cut, λ)/hess_diag[j]
#                 end 
#             else #Must actually perform Armijo line search 
#                 arm = armijo(Xgrp, ygrp, invVgrp, fpars, j, cut, 
#                 hess_diag_untrunc[j], hess_diag[j], cost, p, converged)
#                 fpars = arm.fpars
#                 cost = arm.cost
#                 converged = arm.converged
#             end
#             control.trace > 3 && println(cost) 
#         end
#         control.trace > 3 && println("------------------------")

#         #---Optimization with respect to covariance parameters ----------------------------
#         #------------------------------------------------------------------------------------

         
#         # calculations before the covariance optimization
#         active_set = findall(fpars .== 0)
#         resgrp = resid(XaugGgrp, ygrp, fpars, active_set)
        
#         if ψstr == :ident
            
#         elseif ψstr == :diag

#         else  #ψstr == :sym
        
#         end
            



#     end
# end




end
