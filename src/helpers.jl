"""
    invV!(invVgrp, Zgrp, L, σ²)
Update precision matrices of the responses, by group, by modifying `invVgrp` in place.

# Arguments
- `invVgrp` :: Container for precision matrices of the responses, by group
- `Zgrp` :: Container of random effects design matrices, by group
- `L` :: Parameters for random effect covariance matrix (can be scalar, vector, or lower triangular matrix)
- `σ²` :: Variance of error

"""
function invV!(invVgrp, Zgrp, L, σ²)

    m = size(Zgrp[1])[2]

    if isa(L, Number)
        Ψ = (L^2)I(m)
    elseif isa(L, Vector)
        Ψ = Diagonal(L .^ 2)
    else
        Ψ = L * L'
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
    get_negll(invVgrp, ygrp, XGgrp, β)

Calculate the negative log-likelihod -l(ϕ̃) = -l(β, θ, σ²) = .5(N*log(2π) + log|V| + (y-xβ)'V⁻¹(y-Xβ)) 

# Arguments
- invVgrp :: Vector of length the number of groups, each of whose elements is the precision matrix of the responses within a group
- ygrp :: Vector of vector of responses for each group
- X :: Vector of fixed effect design matrices for each group
- β :: Fixed effects
"""
function get_negll(invVgrp, ygrp, XGgrp, β)

    detV = sum([logabsdet(invV)[1] for invV in invVgrp])
    residgrp = [y - XG * β for (y, XG) in zip(ygrp, XGgrp)]
    quadgrp = [resid' * invV * resid for (resid, invV) in zip(residgrp, invVgrp)]
    Ntot = sum(length.(ygrp))
    return 0.5(Ntot * log(2π) - detV + sum(quadgrp))

end

"""
    get_scad(βpen::Union{Number, Array{Number}}, λ, a = 3.7)

Calculate the SCAD penalty
"""
function get_scad(βpen::Union{Number,Array{Number}}, λ, a = 3.7)

    isa(βpen, Number) && (βpen = [βpen])
    scad = copy(βpen)

    for (j, β) in enumerate(βpen)
        if abs(β) ≤ λ[j]
            scad[j] = λ[j] * abs(β)
        elseif abs(β) ≤ a * λ[j]
            scad[j] = (2 * a * λ[j] * abs(β) - β^2 - λ[j]^2) / (2 * (a - 1))
        else
            scad[j] = (λ[j]^2) * (a + 1) / 2
        end

    end

    return sum(scad)
end

"""
    get_cost(negll, βpen, penalty, λpen, scada = 3.7)

Calculate the objective function
"""
function get_cost(negll, βpen, penalty, λpen, scada = 3.7)

    cost = negll

    if penalty == "lasso"
        cost += sum(λpen .* abs.(βpen))
    elseif penalty == "scad"
        cost += get_scad(βpen, λpen, scada)
    end
    return cost

end


"""
    cov_start(XGgrp, ygrp, Zgrp, β)

Get MLE estimates of L and σ² assuming β are the true fixed effects.

"""
function cov_start(XGgrp, ygrp, Zgrp, β)

    # It can be shown that if we know η = sqrt(Ψ/σ²) = L/σ and β, the MLE for σ² is 
    # given by the below function

    function σ²hat!(
        XGgrp,
        ygrp,
        Zgrp,
        β,
        η,
        invṼgrp = Vector{Matrix{Float64}}(undef, g),
    )::Float64
        invV!(invṼgrp, Zgrp, η, 1)
        residgrp = [y - XG * β for (y, XG) in zip(ygrp, XGgrp)]
        quadgrp = [resid' * invṼ * resid for (resid, invṼ) in zip(residgrp, invṼgrp)]
        σ² = sum(quadgrp) / sum(length.(ygrp))
        return (σ²)
    end

    # We can now profile the likelihood and optimize with respect to γ = log(η) 
    function like(γ, invṼgrp, invVgrp)::Float64
        η = exp(γ)
        σ² = σ²hat!(XGgrp, ygrp, Zgrp, β, η, invṼgrp)
        L = η * sqrt(σ²)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end

    g = length(Zgrp)
    function profile(γ::Float64)::Float64
        like(γ, Vector{Matrix{Float64}}(undef, g), Vector{Matrix{Float64}}(undef, g))
    end

    result = optimize(profile, -10.0, 10.0) #Hard coded interval of optimization for now

    if Optim.converged(result)
        γ = Optim.minimizer(result)
        η = exp(γ)
    else
        error("Convergence for η not reached")
    end

    σ² = σ²hat!(XGgrp, ygrp, Zgrp, β, η)
    L = η * sqrt(σ²)
    return L, σ²

end


"""
    hessian_diag!(XGgrp, invVgrp, active_set, hess, mat_act)

Calculate active_set entries of the diagonal of Hessian matrix for fixed effect parameters 
"""
function hessian_diag!(XGgrp, invVgrp, active_set, hess, mat_act)

    for j in eachindex(invVgrp)
        mat_act[j, :] =
            diag(XGgrp[j][:, active_set]' * invVgrp[j] * XGgrp[j][:, active_set])
    end

    hess[active_set] = sum(mat_act, dims = 1)

end


"""
    special_quad(XG, y, β, j, invVgrp, XGgrp, grp)

Calculate (y-ỹ)' \\* invV \\* X[:,j], where ỹ are the fitted values if we ignored the jth column i.e. XG[:,Not(j)] \\* β[Not(j)]
To improve perforamce, we calculate ỹ at once with the entire dataset.
We then split into groups and calculate (y-ỹ)' \\* invV \\* X[:,j] for each group
"""
function special_quad(XG, y, β, j, invVgrp, XGgrp, grp)

    XGmiss = XG[:, Not(j)]
    βmiss = β[Not(j)]
    resid = y - XGmiss * βmiss


    residgrp = [resid[grp.==group] for group in unique(grp)]

    quads =
        [resid' * invV * XG[:, j] for (resid, invV, XG) in zip(residgrp, invVgrp, XGgrp)]

    return sum(quads)

end

soft_thresh(z, g) = sign(z) * max(abs(z) - g, 0)

"""
    scad_solution(cut, hess, λ, a)

Gets analytical solution for CGD iterate with SCAD penalty when the Hessian hasn't been truncated 
"""
function scad_solution(cut, hess, λ, a)

    if abs(cut) > a * λ
        β = cut / hess
    elseif abs(cut) ≤ 2λ
        β = soft_thresh(cut, λ) / hess
    else
        β = ((a - 1) * cut - sign(cut) * a * λ) / (hess * (a - 2))
    end

    return β
end

"""
    scad_dir(βj::Real, hessj::Real, grad::Real, λj::Real, a::Real)

Calculates descent direction with SCAD penalty
"""
function scad_dir(βj::Real, hessj::Real, grad::Real, λj::Real, a::Real)

    c = βj * hessj - grad

    if c ≤ λj * (hessj + 1)
        d = -(λj + grad) / hessj
    elseif c > λj * a * hessj
        d = -grad / hessj
    else
        d = (-grad * (a - 1) - (a * λj - βj)) / (hessj * (a - 1) - 1)
    end

    return d
end


"""
    armijo!(
        XGgrp,
        ygrp,
        invVgrp,
        β,
        j,
        q,
        cut,
        hessj_untrunc::Real,
        hessj::Real,
        penalty,
        λ,
        a,
        fct_old,
        arm_con,
        control,
    )

Performs Armijo line search to update jth component of β, i.e. β[j]
"""
function armijo!(
    XGgrp,
    ygrp,
    invVgrp,
    β,
    j,
    q,
    cut,
    hessj_untrunc::Real,
    hessj::Real,
    penalty,
    λ,
    a,
    fct_old,
    arm_con,
    control,
)

    βnew = copy(β)

    #Calculate direction
    grad = β[j] * hessj_untrunc - cut

    if j in 1:q
        dir = -grad / hessj
    elseif penalty == "lasso"
        dir = median([(λ[j] - grad) / hessj, -β[j], (-λ[j] - grad) / hessj])
    else ##penalty is SCAD
        dir = scad_dir(β[j], hessj, grad, λ[j], a)
    end

    if dir != 0
        #Calculate Δk
        if j in 1:q
            Δk = dir * grad + control.γ * dir^2 * hessj
        elseif penalty == "lasso"
            Δk =
                dir * grad +
                control.γ * dir^2 * hessj +
                λ[j] * (abs(β[j] + dir) - abs(β[j]))
        else #penalty is SCAD
            Δk =
                dir * grad + control.γ * dir^2 * hessj + get_scad(β[j] + dir, λ[j], a) -
                get_scad(β[j], λ[j], a)
        end

        fct = fct_old
        #Armijo line search
        for l = 0:control.max_armijo

            βnew[j] = β[j] + control.ainit * control.δ^l * dir
            negllnew = get_negll(invVgrp, ygrp, XGgrp, βnew)
            fct_new = get_cost(negllnew, βnew[(q+1):end], penalty, λ[(q+1):end], a)
            addΔ = control.ainit * control.δ^l * control.ρ * Δk

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
    L_ident_update(XGgrp, ygrp, Zgrp, β, σ², var_int, thres)

Update of L for identity covariance structure
"""
function L_scalar_update(XGgrp, ygrp, Zgrp, β, σ², var_int, thres)

    L_lb, L_ub = var_int
    g = length(Zgrp)

    function profile(L::Float64)::Float64
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end
    result = optimize(profile, L_lb, L_ub, show_trace = false)

    Optim.converged(result) || error("Minimization with respect to L failed to converge")
    min = result.minimizer

    if Float64(min) < thres
        L = zero(eltype(min))
        println("L was set to 0 (no group variation)")
    else
        L = min
    end
    return L

end


"""
    L_update!(L::Vector, XGgrp, ygrp, Zgrp, β, σ², s, control)

Update of coordinate s of L for diagonal covariance structure
"""
function L_update!(L::Vector, XGgrp, ygrp, Zgrp, β, σ², s, control)

    g = length(Zgrp)
    #Function to optimize
    function profile(x::Float64)::Float64
        L[s] = x
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end

    x_lb, x_ub = control.var_int
    result = optimize(profile, x_lb, x_ub)

    Optim.converged(result) ||
        error("Minimization with respect to $(s)th coordinate of L failed to converge")
    min = Optim.minimizer(result)

    if min < control.thres
        L[s] = 0
        println("$(s) coordinate of L was set to 0")
    else
        L[s] = min
    end

    return Nothing

end

""" 
    L_update!(L::Matrix, XGgrp, ygrp, Zgrp, β, σ², coords, control)

Update of entry (coords[1], coords[2]) of matrix L, the lower traiangular Choelsky factor of the random effects covariance matrix
"""
function L_update!(L::Matrix, XGgrp, ygrp, Zgrp, β, σ², coords, control)

    #Are we minimizing a covariance parameter or a variance parameters
    int = coords[1] == coords[2] ? control.var_int : control.cov_int
    g = length(Zgrp)
    #Function to optimize
    function profile(x::Float64)::Float64
        L[coords[1], coords[2]] = x
        invVgrp = Vector{Matrix}(undef, g)
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end

    x_lb, x_ub = int
    result = optimize(profile, x_lb, x_ub)

    Optim.converged(result) ||
        error("Minimization with respect to $(coords) entry of L failed to converge")
    min = Optim.minimizer(result)
    if coords[1] == coords[2] && min < control.thres
        L[coords[1], coords[2]] = 0
        println("$(coords) entry of L was set to 0")
    else
        L[coords[1], coords[2]] = min
    end
    #println("L is now $L and log-likelihood is $(Optim.minimum(result))")

    return Nothing
end


"""
    σ²update(XGgrp, ygrp, Zgrp, β, L, var_int)

Update the variance component parameter σ² using profile likelihood optimization.

# Arguments
- `XGgrp`::Vector{Matrix} Vector with fixed effect design matrices for each group.
- `ygrp::Vector{Vector}`: Response vectors.
- `Zgrp::Vector{Matrix}`: Design matrices for the random effects.
- `β::Vector{Real}`: Fixed effects coefficients.
- `L::Union{Real, Vector, Matrix}`: Cholesky factorizations of the random effects covariance matrices.
- `var_int::Tuple{Real, Real}`: Lower and upper bounds for the variance component.

# Returns
- `σ²::Float64`: Updated value of the variance component.

"""
function σ²update(
    XGgrp::Vector{Matrix},
    ygrp::Vector{Vector},
    Zgrp::Vector{Matrix},
    β::Vector{Real},
    L::Union{Real,Vector,Matrix},
    var_int::Tuple{Real,Real},
)

    #Decision variable will be σ² rather than σ
    function profile(σ²::Float64)::Float64
        invVgrp = Vector{Matrix}(undef, length(Zgrp))
        invV!(invVgrp, Zgrp, L, σ²)
        get_negll(invVgrp, ygrp, XGgrp, β)
    end

    x_lb, x_ub = var_int
    result = optimize(profile, x_lb^2, x_ub^2) #Square bounds because decision variable is σ²

    Optim.converged(result) || error("Minimization with respect to σ² failed to converge")

    return Optim.minimizer(result)

end
