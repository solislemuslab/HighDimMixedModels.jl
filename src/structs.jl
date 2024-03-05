"""
Algorithm Hyper-parameters

- tol :: Convergence tolerance
- seed :: Random seed for cross validation for estimating initial fixed effect parameters using Lasso
- trace :: Integer. 1 prints no output, 2 prints issues, and 3 prints the objective function values during the algorithm and issues
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
    tol::Float64 = 1e-4
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
    var_int::Tuple{Float64,Float64} = (0, 100)
    cov_int::Tuple{Float64,Float64} = (-5, 5)
    optimize_method::Symbol = :Brent
    thres::Float64 = 1e-4
end

"""
Fitted model object 
"""
mutable struct HDMModel 
    data::NamedTuple
    weights::Vector
    init_coef::NamedTuple
    init_log_like::Float64
    init_objective::Float64
    init_nz::Int
    penalty::String
    λ::Real 
    scada::Real 
    σ²::Real
    L::Matrix
    fixef::Vector
    ranef::Vector
    fitted::Vector
    resid::Vector
    log_like::Float64
    objective::Float64
    npar::Int
    nz::Int
    deviance::Float64
    arm_con::Int
    num_arm::Int
    aic::Float64
    bic::Float64
    iterations::Int
    ψstr::String
    ψ::Matrix
    control::Control
end
