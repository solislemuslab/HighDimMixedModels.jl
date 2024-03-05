# HighDimMixedModels

HighDimMixedModels.jl is a package for fitting penalized linear mixed-effect models. It is a Julia implementation of the algorithm described in

Schelldorfer, Jürg, Peter Bühlmann, and SARA VAN DE GEER. "Estimation for high‐dimensional linear mixed‐effects models using ℓ1‐penalization." Scandinavian Journal of Statistics 38.2 (2011): 197-214.

Two options for penalties are provided, the original LASSO and the smoothly clipped absolute deviation (SCAD) penalty described in

Ghosh, Abhik and Magne Thoresen. “Non-concave penalization in linear mixed-effect models and regularized selection of fixed effects.” AStA Advances in Statistical Analysis 102 (2016): 179 - 210.


# Example 
The [cognitive dataset](data/cognitive.csv) contains data from a [study](https://www.sciencedirect.com/science/article/pii/S0022316623025622) of the effect of an intervention in school lunches among schools in Kenya, accessed via the R package [`splmm`](https://cran.r-project.org/web/packages/splmm/index.html), which fits high dimensional mixed effect models using the same algorithm. The data is longitudinal with measurements of students taken at multiple time points, and we will assign each student a random intercept.

Load data into Julia 
```
using CSV
using DataFrames
using CategoricalArrays
cog_df = CSV.read("data/cognitive.csv", DataFrame)
# form categorical variable
cog_df.schoolid = categorical(cog_df.schoolid)
cog_df.treatment = categorical(cog_df.treatment, levels=["control", "calorie", "meat", "milk"]) 
```

Form the model matrices 
```
f = @formula(ravens ~ 1 + schoolid + treatment + year + sex + age_at_time0 +
                      height + weight + head_circ + ses + mom_read + mom_write + mom_edu)
mf = ModelFrame(f, cog_df)
mm = ModelMatrix(mf).m
X = mm[:, 1:1] # Covariates whose coefficients will not be penalized, in this case, just the intercept
G = mm[:, 2:end] # High dimensional covariates whose coefficients will be penalized
```

Get the cluster (in this case, student) ids, the response, and fit the model with the main function `hdmm`
```
student_id = cog_df.id
y = cog_df.ravens
fit = hdmm(X, G, y, student_id)
```

Note that this fits a model with the SCAD penalty by default. To apply the LASSO penalty, simply specify `penalty = "lasso"` in the call to fit the model. 

By default, the features that are assigned random slopes are all those that appear as columns in the matrix `X`, i.e. those whose coefficients in the model are not penalized. In the above code, `X` just contains a single constant column, so we are fitting a model with just random intercepts. It's possible, however, to include additional columns in `X`. If you do so, each corresponding feature will receive a random slopes, by default. If you wish to include a feature whose coefficient is not penalized, but do not wish to assign this feature a random slope, then you can specify the argument `Z` in the call to `hdmm` to be a matrix whose columns contain only the variables in `X` that you wish to assign random slopes.

As HighDimMixedModels.jl implements some of the abstraction for statistical models provided by [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/tree/master), we can inspect the model using common API functions. For example, to get all estimated fixed effect coefficients:
```
coef(fit)
```
To print a table with only the estimated fixed effect coefficients that are non-zero:
```
coeftable(fit, coefnames(mf))
```
To view the fitted and residuals:
```
fitted(fit)
residuals(fit)
```
Note that these fitted values and residuals take into account the random effects by incorporating the best prediction of these random effects (BLUPs) for each student into the predictions.

