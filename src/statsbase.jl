# Extending generics from StatsAPI to an HDMModel object
# These are functions for inspecting a fitted model

coef(fit::HDMModel) = fit.fixef

function coeftable(fit::HDMModel, names::Vector{String}=string.(1:length(fit.fixef))) 
    fixef = coef(fit)
    nz = findall(fixef .!= 0)
    StatsBase.CoefTable([fixef[nz]], ["Estimate"], names[nz])
end

deviance(fit::HDMModel) = fit.deviance
loglikelihood(fit::HDMModel) = fit.log_like
nobs(fit::HDMModel) = length(fit.fitted)
residuals(fit::HDMModel) = fit.resid
fitted(fit::HDMModel) = fit.fitted
aic(fit::HDMModel) = fit.aic
bic(fit::HDMModel) = fit.bic





