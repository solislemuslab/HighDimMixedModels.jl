# Extending generics from StatsAPI to an HDMModel object
# These are functions for inspecting a fitted model

"""
    coef(fit::HDMMModel)
Retrieves all estimated fixed effect coefficients
"""
coef(fit::HDMModel) = fit.fixef

"""
    coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef)))

Returns a table of the *selected* coefficients, i.e. those not set to 0, from the model.

# Arguments
- `fit::HDMModel`: A fitted model.
- `names::Vector{String}`: Names of the all the coefficients in the model (not just those selected), defaults to integer names 

# Returns
A `StatsBase.CoefTable` object.
"""
function coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef))) 
    fixef = coef(fit)
    nz = findall(fixef .!= 0)
    StatsBase.CoefTable([fixef[nz]], ["Estimate"], names[nz])
end

"""
    loglikelihood(fit::HDMModel)
Log-likelihood of the model at the estimated parameters
"""
loglikelihood(fit::HDMModel) = fit.log_like

"""
    deviance(fit::HDMModel)
-2*loglikelihood of the model at the estimated parameters
"""
deviance(fit::HDMModel) = fit.deviance

"""
    nobs(fit::HDMModel)
Number of observations used in fitting the model
"""
nobs(fit::HDMModel) = length(fit.fitted)

"""
    fitted(fit::HDMModel)
Accounts for the random effects in generating predictions
"""
fitted(fit::HDMModel) = fit.fitted

"""
    residuals(fit::HDMModel)
Accounts for the random effects in generating predictions
"""
residuals(fit::HDMModel) = fit.resid

"""
    aic(fit::HDMModel)
The Akaike Information Criterion is equal to the deviance plus 2 times the number of parameters in the model
"""
aic(fit::HDMModel) = fit.aic

"""
    bic(fit::HDMModel)
The Bayesian Information Criterion is equal to the deviance plus the log of the number of observations times the number of parameters in the model
"""
bic(fit::HDMModel) = fit.bic





