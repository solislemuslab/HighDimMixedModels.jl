# Extending generics from StatsAPI to an HDMModel object
# These are functions for inspecting a fitted model

"""
    coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef)))

Return a table of the *selected* coefficients, i.e. those not set to 0, from the model.

# Arguments
- `fit::HDMModel`: A fitted model.
- `names::Vector{String}`: Names of the all the coefficients in the model (not just those selected), defaults to integer names 

# Returns
A `StatsBase.CoefTable` object.
"""
function coeftable(fit::HDMModel, names::Vector{String} = string.(1:length(fit.fixef)))
    fixef = coef(fit)
    nz = findall(fixef .!= 0)
    StatsBase.CoefTable([fixef[nz]], ["Estimate"], names[nz])
end

coef(fit::HDMModel) = fit.fixef
loglikelihood(fit::HDMModel) = fit.log_like
deviance(fit::HDMModel) = fit.deviance
nobs(fit::HDMModel) = length(fit.fitted)
aic(fit::HDMModel) = fit.aic
bic(fit::HDMModel) = fit.bic

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
