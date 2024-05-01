
## Fitting model

To fit a regularized mixed effect model with the package, simply run
```
fit = hdmm(X, G, y, student_id)
```

This function has a number of keyword arguments that can be specified to modify the defaults--see the [documentation](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/#HighDimMixedModels.hdmm). For example, by default, we fit a model with the SCAD penalty, which produces less bias. To apply the LASSO penalty, specify `penalty = "lasso"` in the call. Also note that the default value of $\lambda$ (the hyperparameter that controls the degree of penalization) is 10. Since the default (unless `standardize = false`) is to standardize the design matrices before running the algorithm, this is how much the coefficients of the standardized predictors are penalized. In practice, you can and should try fitting the model for several different choices of $\lambda$ and choose the fit that leads to the lowest `fit.BIC`.

By default, the features that are assigned random slopes are all those that appear as columns in the matrix `X`, i.e. those features whose coefficients are not penalized. If you wish to include a feature whose coefficient is not penalized, but do not wish to assign this feature a random slope, then you can specify the argument `Z` in the call to `hdmm` to be a matrix whose columns contain only the variables in `X` that you wish to assign random slopes.

## Inspecting model

The object `fit` which is returned by `hdmm()` is a struct with fields providing all relevant information about the model fit. These can be accessed using the `dot` notation, e.g. `fit.fixef` to retrieve all the fixed effect estimates (including those set to 0) and `fit.log_like` to get the log likelihood at the estimates. To print all the fields stored in the object, you can type `fit.` followed by the tab key.

We also implement several common extraction functions from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/tree/master), such as `residuals(fit)` and `fitted(fit)`. Note that these fitted values and residuals take into account the random effects by incorporating the best prediction of these random effects (BLUPs) for each student into the predictions. 

To print a table with only the selected coefficients (i.e. those that are not set to 0), use the function `coeftable()`. The names of these variables will appear alongside their estimates if you pass them as a second argument:
```@example cog
coeftable(fit, coefnames(model_frame))
```

## Plotting model

Here, we show how to plot the observed scores and our model's predictions for five different students over time:
```@example cog
using Plots
mask = student_id .== 1
plot(cog_df.year[mask], cog_df.ravens[mask], seriestype = :scatter, label = "student 1", color = 1 )
plot!(cog_df.year[mask], fitted(fit)[mask], seriestype = :line, color = 1, linestyle = :dash, linewidth = 3, label = "")
for i in [2,4,5,6]
    mask = student_id .== i
    # add student to plot
    plot!(cog_df.year[mask], cog_df.ravens[mask], seriestype = :scatter, label = "student $i", color = i)
    plot!(cog_df.year[mask], fitted(fit)[mask], seriestype = :line, color = i, linestyle = :dash, linewidth = 3, label = "")
end
plot!(legend=:outerbottom, legendcolumns=3, xlabel = "Year", ylabel = "Ravens Score")
```
