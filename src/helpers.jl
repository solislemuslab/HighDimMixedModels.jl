using LinearAlgebra
using Random
using Lasso #Needed to get initial estimates of the fixed effects
using Optim #Needed for univariate optimization in coordinated gradient descent algorithm
using InvertedIndices #Allows negative indexing, like in R

"""
Updates precision matrices of the responses, by group

ARGUMENTS
- invVgrp :: Container for precision matrices of the responses, by group
- Zgrp :: Container of random effects design matrices, by group
- L :: Parameters for random effect covariance matrix (can be scalar, vector, or lower triangular matrix)
- σ² :: Variance of error

OUTPUT
- invVgrp :: List of length the number of groups, each of whose elements is the covariance matrix of the responses within a group
"""
function invV!(invVgrp, Zgrp, L, σ²)
    
    m = size(Zgrp[1])[2]
    
    if isa(L, Number)
        Ψ = (L^2)I(m)
    elseif isa(L, Vector)
        Ψ = Diagonal(L.^2)
    else
        Ψ = L*L' 
    end
    
    for (i, Zᵢ) in enumerate(Zgrp)
        nᵢ = size(Zᵢ)[1]
        Vᵢ = Zᵢ * Ψ * Zᵢ' + σ² * I(nᵢ)
        # Need to make symmetric: because of floating point arithmetic, 
        # cholesky will fail to recognize the matrix as symmetric
        invVgrp[i] = inv(cholesky(Symmetric(Vᵢ)))
    end

end

"""
Calculates the negative log-likelihod 
-l(ϕ̃) = -l(β, θ, σ²) = .5(Ntot*log(2π) + log|V| + (y-xβ)'V⁻¹(y-Xβ)) 

ARGUMENTS
- invVgrp :: Vector of length the number of groups, each of whose elements is the precision matrix of the responses within a group
- ygrp :: Vector of vector of responses for each group
- X :: Vector of fixed effect design matrices for each group
- β :: Fixed effects

OUTPUT
- Value of the negative log-likelihood
"""
function get_negll(invVgrp, ygrp, XGgrp, β)

    detV = sum([logabsdet(invV)[1] for invV in invVgrp])
    residgrp = [y-XG*β for (y, XG) in zip(ygrp, XGgrp)]
    quadgrp = [resid'*invV*resid for (resid, invV) in zip(residgrp, invVgrp)]
    Ntot = sum(length.(ygrp))
    return .5(Ntot*log(2π) - detV + sum(quadgrp)) 

end

"""
Calculates the SCAD penalty
"""
function get_scad(βpen, λ, a=3.7) 
    
    isa(βpen, Number) && (βpen = [βpen])
    scad = copy(βpen)

    for (j, β) in enumerate(βpen)
        if abs(β) ≤ λ[j]
            scad[j] = λ[j]*abs(β)
        elseif abs(β) ≤ a*λ[j]
            scad[j] = (2*a*λ[j]*abs(β) - β^2 - λ[j]^2)/(2*(a-1))
        else 
            scad[j] = (λ[j]^2)*(a+1)/2
        end
        
    end

    return sum(scad)
end

"""
Calculates the objective function
"""
function get_cost(negll, βpen, penalty, λpen, scada=3.7)
    
    cost = negll
    
    if penalty == "lasso"
        cost += sum(λpen.*abs.(βpen))
    elseif penalty == "scad"
        cost += get_scad(βpen, λpen, scada)
    end
    return cost

end 


"""
Finds an initial value for the variance and covariance parameters 

ARGUMENTS
- XGgrp :: Vector of fixed effect design matrices for each group
- ygrp :: Vector of vector of responses for each group
- Zgrp :: Vector of random effects design matrix for each group
- β :: Initial iterate for fixed effect parameter vector (computed with Lasso ignoring group structure)

OUTPUT
- Assuming β is true fixed effect vector, MLE estimate of scalar L and scalar σ² as tuple (L, σ²)

"""
function cov_start(XGgrp, ygrp, Zgrp, β)
    
    # It can be shown that if we know η = sqrt(Ψ/σ²) and β, the MLE for σ² is 
    # given by the below function

    function σ²hat!(XGgrp, ygrp, Zgrp, β, η, invṼgrp) 
        invV!(invṼgrp, Zgrp, η, 1)
        residgrp = [y-XG*β for (y, XG) in zip(ygrp, XGgrp)]
        quadgrp = [resid'*invṼ*resid for (resid, invṼ) in zip(residgrp, invṼgrp)]
        σ² = sum(quadgrp)/sum(length.(ygrp))
        return(σ²)
    end
    
    # We can now profile the likelihood and optimize with respect to η 
    function like(η, invṼgrp, invVgrp)
        σ² = σ²hat!(XGgrp, ygrp, Zgrp, β, η, invṼgrp)
        L = η*sqrt(σ²)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β) 
    end

    g = length(Zgrp)
    profile(η) = like(η, Vector{Matrix}(undef, g), Vector{Matrix}(undef, g))

    result = optimize(profile, 1.0e-6, 1.0e6) #will need to fix
    
    if Optim.converged(result)
        η = Optim.minimizer(result)
    else
        error("Convergence for η not reached")
    end

    σ² = σ²hat!(XGgrp, ygrp, Zgrp, β, η, Vector{Matrix}(undef, g))
    L = η*sqrt(σ²)
    return L, σ²

end


"""
Calculates active_set entries of the diagonal of Hessian matrix for fixed effect parameters 
"""
function hessian_diag!(XGgrp, invVgrp, active_set, hess, mat_act)

    for j in eachindex(invVgrp)
        mat_act[j,:] = diag(XGgrp[j][:,active_set]'*invVgrp[j]*XGgrp[j][:,active_set])
    end

    hess[active_set] = sum(mat_act, dims=1)

end


"""
Calculates (y-ỹ)'*(invV)*X[:,j], where ỹ are the fitted values if we ignored the jth column i.e. XG[:,Not(j)]*β[Not(j)]
To improve perforamce, we calculate ỹ with the entire dataset 
We then split into groups and calculate (y-ỹ)'*(invV)*X[:,j] for each group
"""
function special_quad(XG, y, β, j, invVgrp, XGgrp, grp) 
    
    XGmiss = XG[:,Not(j)]
    βmiss = β[Not(j)]
    resid = y - XGmiss*βmiss


    residgrp = [resid[grp .== group] for group in unique(grp)]

    quads = [resid'*invV*XG[:,j] for (resid, invV, XG) in zip(residgrp, invVgrp, XGgrp)]

    return sum(quads)

end

"""
Soft Threshold
"""
soft_thresh(z,g) = sign(z)*max(abs(z)-g,0)

"""
Gets analytical solution for CGD iterate with SCAD penalty when the Hessian hasn't been truncated 
"""
function scad_solution(cut, hess, λ, a) 
    
    if abs(cut) > a*λ
        β = cut/hess
    elseif abs(cut) ≤ 2λ
        β = soft_thresh(cut, λ)/hess
    else 
        β = ((a-1)*cut - sign(cut)*a*λ)/(hess*(a-2))
    end

    return β
end

"""
Calculates descent direction with SCAD penalty
"""
function scad_dir(βj::Real, hessj::Real, grad::Real, λj::Real, a::Real)

    c = βj*hessj-grad

    if c ≤ λj*(hessj + 1)
        d = -(λj + grad)/hessj
    elseif c > λj*a*hessj
        d = -grad/hessj
    else
        d = (-grad*(a-1)-(a*λj - βj))/(hessj*(a-1)-1)
    end
    
    return d
end


"""
Armijo Rule
"""
function armijo!(XGgrp, ygrp, invVgrp, β, j, q, cut, 
    hessj_untrunc::Real, hessj::Real, penalty, 
    λ, a, fct_old, arm_con, control)
    
    βnew = copy(β)
   
    #Calculate direction
    grad = β[j]*hessj_untrunc - cut
    
    if j in 1:q
        dir = -grad/hessj
    elseif penalty == "lasso"
        dir = median([(λ[j] - grad)/hessj, -β[j], (-λ[j] - grad)/hessj])
    else ##penalty is SCAD
        dir = scad_dir(β[j], hessj, grad, λ[j], a)
    end

    if dir != 0
        #Calculate Δk
        if j in 1:q
            Δk = dir*grad + control.γ*dir^2*hessj
        elseif penalty == "lasso"
            Δk = dir*grad + control.γ*dir^2*hessj + λ[j]*(abs(β[j]+dir)-abs(β[j]))
        else #penalty is SCAD
            Δk = dir*grad + control.γ*dir^2*hessj + get_scad(β[j] + dir, λ[j], a) - get_scad(β[j], λ[j], a)
        end

        fct = fct_old
        #Armijo line search
        for l in 0:control.max_armijo

            βnew[j] = β[j] + control.ainit*control.δ^l*dir
            negllnew = get_negll(invVgrp, ygrp, XGgrp, βnew)
            fct_new = get_cost(negllnew, βnew[(q+1):end], penalty, λ[(q+1):end], a)
            addΔ = control.ainit*control.δ^l*control.ρ*Δk

            if fct_new <= fct + addΔ
                β[j] = βnew[j]
                fct = fct_new
                return (fct = fct, arm_con = arm_con)
            end
            if l == control.max_armijo
                control.trace > 1 && @warn "Armijo for coordinate $(j) of β not successful" 
                arm_con += 1 
                return (fct = fct, arm_con = arm_con)
            end
        end
    end

end

""" 
Update of L for identity covariance structure
""" 
function L_ident_update(XGgrp, ygrp, Zgrp, β, σ², var_int, thres)

    L_lb, L_ub = var_int
    g = length(Zgrp)
    
    function profile(L) 
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end
    result = optimize(profile, L_lb, L_ub, show_trace = false) 
    
    Optim.converged(result) || error("Minimization with respect to L failed to converge")
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
function L_diag_update!(L, XGgrp, ygrp, Zgrp, β, σ², s, var_int, thres)
    
    g = length(Zgrp)
    #Function to optimize
    function profile(x)
        L[s] = x 
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β) 
    end

    x_lb, x_ub = var_int
    result = optimize(profile, x_lb, x_ub) 

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
function L_sym_update!(L, XGgrp, ygrp, Zgrp, β, σ², coords, var_int, cov_int, thres)
    
    #Are we minimizing a covariance parameter or a variance parameters
    int = coords[1]==coords[2] ? var_int : cov_int 
    g = length(Zgrp)
    #Function to optimize
    function profile(x)
        L[coords[1], coords[2]] = x 
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end

    x_lb, x_ub = int
    result = optimize(profile, x_lb, x_ub)

    Optim.converged(result) || error("Minimization with respect to $(coords) entry of L failed to converge")
    min = Optim.minimizer(result)
    if coords[1]==coords[2] && min < thres 
        L[coords[1], coords[2]] = 0
        println("$(coords) entry of L was set to 0")
    else
        L[coords[1], coords[2]] = min
    end
    #println("L is now $L and log-likelihood is $(Optim.minimum(result))")

    return Nothing
end

"""
Update of σ²
"""
function σ²update(XGgrp, ygrp, Zgrp, β, L, var_int)
   
    #Decision variable will be σ² rather than σ
    function profile(σ²) 
        invVgrp = Vector{Matrix}(undef, length(Zgrp))
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end
    
    x_lb, x_ub = var_int
    result = optimize(profile, x_lb^2, x_ub^2) #Square bounds because decision variable is σ²
    
    Optim.converged(result) || error("Minimization with respect to σ² failed to converge")

    return Optim.minimizer(result)

end

