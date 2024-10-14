"""
    Control

Hyperparameters for the coordinate descent algorithm

# Fields
- `tol`: Small positive number, default is 1e-4, providing convergence tolerance
- `seed`: Random seed, default 770. Note that the only randomness in the algorithm is during the initialization of fixed effect parameters (for the data splits in the cross validation)
- `trace`: Bool, default `false`. If `true`, prints cost and size of active set over the course of the algorithm.
- `max_iter`: Integer, default 1000, giving maximum number of iterations in the coordinate gradient descent.
- `max_armijo`: Integer, default 20, giving the maximum number of steps in the Armijo algorithm. If the maximum is reached, algorithm doesn't update current coordinate and proceeds to the next coordinate
- `act_num`: Integer, default 5. We will only update all of the fixed effect parameters every `act_num` iterations. Otherwise, we update only the parameters in the current active set.
- `a₀`: a₀ in the Armijo step, default 1.0. See Schelldorfer et al. (2010) for details about this and the next five fields.
- `δ`: δ in the Armijo step, default 0.1. 
- `ρ`: ρ in the Armijo step, default 0.001. 
- `γ`: γ in the Armijo step, default 0.0. 
- `lower`: Lower bound for the Hessian, default 1e-6. 
- `upper`: Upper bound for the Hessian, default 1e8.
- `var_int`: Tuple with bounds of interval on which to optimize when updating a diagonal entry of L, default (0, 100). See Optim.jl in section "minimizing a univariate function on a bounded interval"
- `cov_int`: Tuple with bounds of interval on which to optimize the when updating a non-diagonal entry of L, default (-50, 50). See Optim.jl in section "minimizing a univariate function on a bounded interval"
- `optimize_method`: Symbol denoting method for performing the univariate optimization, either :Brent or :GoldenSection, default is :Brent
- `thres`: If an update to a diagonal entry of L is smaller than `thres`, the parameter is set to 0
"""
@with_kw mutable struct Control
    tol::Real = 1e-4
    seed::Int = 770
    trace::Bool = false
    max_iter::Int = 1000
    max_armijo::Int = 20
    act_num::Int = 5
    ainit::Real = 1.0
    δ::Real = 0.1
    ρ::Real = 0.001
    γ::Real = 0.0
    lower::Real = 1e-6
    upper::Real = 1e8
    var_int::Tuple{Real,Real} = (0, 100)
    cov_int::Tuple{Real,Real} = (-50, 50)
    optimize_method::Symbol = :Brent
    thres::Real = 1e-4
end

"""
    HDMModel

Results of a fitted model

# Fields
- `data`: NamedTuple containing the input data used for fitting the model
- `weights`: Vector of penalty weights used in the model
- `init_coef`: NamedTuple containing the initial coefficient values
- `init_log_like`: Initial log-likelihood value
- `init_objective`: Initial objective function value
- `init_nz`: Number of non-zero components in the initial estimate of fixed effects
- `penalty`: String indicating the type of penalty used in the model
- `standardize`: Boolean indicating whether the input data was standardized
- `λ`: Regularization hyperparameter 
- `scada`: Hyperparameter relevant to the scad penalty
- `σ²`: Estimated variance parameter
- `L`: Lower triangular matrix representing the Cholesky factor of the random effect covariance matrix
- `fixef`: Vector of estimated fixed effects
- `ranef`:  vector of g vectors, each of length m, holding random effects BLUPs for each group
- `fitted`: Vector of fitted values, including random effects
- `resid`: Vector of residuals, including random effects
- `log_like`: Log-likelihood value at convergence
- `objective`: Objective function value at convergence
- `npar`: Total number of parameters in the model
- `nz`: Number of non-zero fixed effects
- `deviance`: Deviance value
- `num_arm`: Number of times `armijo!` needed to be called 
- `arm_con`: Number of times the Armijo algorithm failed to converge
- `aic`: Akaike Information Criterion
- `bic`: Bayesian Information Criterion
- `iterations`: Number of iterations performed
- `ψstr`: Assumed structure of the random effect covariance matrix
- `ψ`: Estimated random effect covariance matrix, i.e. L * L'
- `control`: Control object containing hyperparameters that were used for the coordinate descent algorithm
"""
struct HDMModel
    data::Union{NamedTuple, Nothing} # allow for discarding of data to save space
    weights::Union{Vector, Nothing} # allow for no weights
    init_coef::NamedTuple
    init_log_like::Float64
    init_objective::Float64
    init_nz::Int
    penalty::String
    standardize::Bool
    λ::Real
    scada::Real
    σ²::Real
    L::Matrix
    fixef::Vector 
    ranef::Union{Vector, Nothing} # allow for discarding of random effects to save space
    fitted::Union{Vector, Nothing} # allow for discarding of fitted values to save space
    resid::Union{Vector, Nothing} # allow for discarding of residuals to save space
    log_like::Float64
    objective::Float64
    npar::Int
    nz::Int
    deviance::Float64
    num_arm::Int
    arm_con::Int
    aic::Float64
    bic::Float64
    iterations::Int
    ψstr::String
    ψ::Matrix
    control::Control
end


function Base.show(io::IO, ::MIME"text/plain", obj::HDMModel)
    println(io, "HDMModel fit with $(length(obj.fitted)) observations")
    println(io, "Log-likelihood at convergence: $(round(obj.log_like, sigdigits=5))")
    println(io, "Random effect covariance matrix:")
    show(io, "text/plain", obj.ψ)
    println(io)
    println(io, "Estimated $(obj.nz) non-zero fixed effects:")
    show(io, coeftable(obj))
    println(io)
    println(io, "Estimated σ²: $(round(obj.σ², sigdigits=5))")
end
