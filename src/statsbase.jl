# Classes and functions for model inspection
StatsBase.coef(fit::HDMModel) = fit.fixef

function StatsBase.coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef))) 
    fixef = StatsBase.coef(fit)
    nz = findall(fixef .!= 0)
    StatsBase.CoefTable([fixef[nz]], ["Estimate"], names[nz])
end

StatsBase.deviance(fit::HDMModel) = fit.deviance
StatsBase.loglikelihood(fit::HDMModel) = fit.log_like
StatsBase.nobs(fit::HDMModel) = length(fit.fitted)
StatsBase.residuals(fit::HDMModel) = fit.resid
StatsBase.fitted(fit::HDMModel) = fit.fitted
StatsBase.aic(fit::HDMModel) = fit.aic
StatsBase.bic(fit::HDMModel) = fit.bic
