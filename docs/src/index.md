# HighDimMixedModels.jl

HighDimMixedModels.jl is a package for fitting regularized linear mixed-effect models to high dimensional, clustered data. It is a Julia implementation of the estimation approach in

>Schelldorfer, J., Bühlmann, P., & DE GEER, S. V. (2011). Estimation for high‐dimensional linear mixed‐effects models using ℓ1‐penalization. Scandinavian Journal of Statistics, 38(2), 197-214.

Two options for penalties are provided, the original LASSO and the smoothly clipped absolute deviation (SCAD) penalty described in

>Ghosh, A., & Thoresen, M. (2018). Non-concave penalization in linear mixed-effect models and regularized selection of fixed effects. AStA Advances in Statistical Analysis, 102, 179-210.

## Quick Start

The [cognitive dataset](data/cognitive.csv) contains data from a [study](https://www.sciencedirect.com/science/article/pii/S0022316623025622) of the effect of an intervention in school lunches among schools in Kenya, accessed via the R package [`splmm`](https://cran.r-project.org/web/packages/splmm/index.html). The data is longitudinal with measurements of students' performance on various tests taken at different points in time. We will fit a model with random intercepts and random growth slopes for each student. Note that while this is a low-dimensional example (``p < n``), the algorithm that this package implements was designed and tested with the high dimensional use-case (``p > n``) in mind.

First, we load the data into Julia and form a categorical variable for the treatment in the study, which was the type of lunch served (assigned at the school level).
```@example cog
using CSV
using DataFrames
using CategoricalArrays
cog_df = CSV.read("data/cognitive.csv", DataFrame)
# form categorical variable for treatment
cog_df.treatment = categorical(cog_df.treatment, levels=["control", "calorie", "meat", "milk"])
nothing #hide
```

Next we form model matrices with the help of the [`StatsModels`](https://juliastats.org/StatsModels.jl/stable/formula/#The-@formula-language) formula syntax:
```@example cog
using StatsModels
f = @formula(ravens ~ 1 + treatment + year + sex + age_at_time0 +
                      height + weight + head_circ + ses + mom_read + mom_write + mom_edu)
model_frame = ModelFrame(f, cog_df)
model_mat = ModelMatrix(model_frame).m
nothing #hide
```

We form two model matrices. One is low dimensional and includes only the columns that will have associated random effects and the other is higher dimensional and includes the many features whose effects will be regularized. 
```@example cog
X = model_mat[:, 1:2] # Non-penalized, random effect columns (one for intercept, and the other for year)
G = model_mat[:, 3:end] # High dimensional set of covariates whose effects are regularized
nothing #hide
```
Finally, we get the cluster (in this case, student) ids and the response, the students' Ravens test scores, and fit the model with the main function from the package, `hdmm`:

```@example cog
using HighDimMixedModels
student_id = cog_df.id
y = cog_df.ravens
fit = hdmm(X, G, y, student_id)
```

Note that by default, we fit a model with the SCAD penalty. To apply the LASSO penalty, simply specify `penalty = "lasso"` in the call to fit the model. 

By default, the features that are assigned random slopes are all those that appear as columns in the matrix `X`, i.e. those whose coefficients in the model are not penalized. In the above code, `X` just contains a single constant column, so we are fitting a model with just random intercepts. It's possible, however, to include additional columns in `X`. If you do so, each corresponding feature will receive a random slopes, by default. If you wish to include a feature whose coefficient is not penalized, but do not wish to assign this feature a random slope, then you can specify the argument `Z` in the call to `hdmm` to be a matrix whose columns contain only the variables in `X` that you wish to assign random slopes.

We can inspect the model using common extraction functions from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/tree/master). For example, to get the residuals and fitted values,
```@repl cog
residuals(fit)
fitted(fit)
```

Note that these fitted values and residuals take into account the random effects by incorporating the best prediction of these random effects (BLUPs) for each student into the predictions.

To print a table with the names of the selected variables and their estimated coefficients:
```@example cog
coeftable(fit, coefnames(model_frame))
```

Here, we plot the observed scores and our model's predictions for five different students over time:
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

## Function Documentation
```@autodocs
Modules = [HighDimMixedModels]
```