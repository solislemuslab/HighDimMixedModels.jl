# Extending generics from StatsAPI to HDMModel
# These are functions for inspecting a fitted model

StatsAPI.coef(fit::HDMModel) = fit.fixef

function StatsAPI.coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef))) 
    fixef = StatsBase.coef(fit)
    nz = findall(fixef .!= 0)
    StatsBase.CoefTable([fixef[nz]], ["Estimate"], names[nz])
end

StatsAPI.deviance(fit::HDMModel) = fit.deviance
StatsAPI.loglikelihood(fit::HDMModel) = fit.log_like
StatsAPI.nobs(fit::HDMModel) = length(fit.fitted)
StatsAPI.residuals(fit::HDMModel) = fit.resid
StatsAPI.fitted(fit::HDMModel) = fit.fitted
StatsAPI.aic(fit::HDMModel) = fit.aic
StatsAPI.bic(fit::HDMModel) = fit.bic
