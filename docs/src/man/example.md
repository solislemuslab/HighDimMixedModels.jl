# Examples

## Simulated data

We first generate two design matrices. The first, `X`, will be low dimensional and consist of the predictors that will have associated random effects--this includes a column of constant 1s so that we include random intercepts. The second, `G`, is high dimensional and includes predictors whose effects we penalize.

In addition, we generate a sparse vector of regression coefficient, $\beta$, whose non-zero components are centered at 0 and have a standard deviation of 1. 
```@example sim
using Random
Random.seed!(420)
g, n, p, q = 100, 5, 1000, 4 # #groups, #samples per group, #features, #features with random effect (including random intercept)
N = n*g  # Total sample size
X = [ones(N) randn(N, q-1)]
G = randn(N, p-q)
XG = [X G] 
sd_signal = 1
β = [sd_signal .* randn(10); zeros(p-10)]
nothing
```

Now we generate the response from these design matrices
```@example sim
using Distributions
using LinearAlgebra
# Generate group assignments
gr = string.( vcat( [fill(i, n) for i in 1:g]... ) )

# Generate random effects
ψ = Diagonal(ones(q)) #random effects all have unit sd
dist_b = MvNormal(zeros(q), ψ) 
b = rand(dist_b, g)

# Generate response
y_fixed = XG * β 
y = Vector{Float64}(undef, N)
for (i, group) in enumerate(unique(grp))
    group_ind = (gr .== group)
    nᵢ = sum(group_ind)
    Xᵢ = X[group_ind,:]
    bᵢ = b[:,i]
    yᵢ = Zᵢ*bᵢ + randn(nᵢ)
    y[group_ind] = y_fixed[group_ind] + yᵢ
end
nothing
```
We can now fit the model. We prepare design matrices, X and G, that the function requires. 
X contains the variables whose effect we do not wish to penalize and a column for the intercept. These ar G is the high dimensional matrix with all the variables whose effect we do want to penalize. 
By default, the variables in X (including the intercept) are assigned a random effect and by default, the assumed covariance structure of the random effects is diagonal. After some experimentation, we can set the penalty $\lambda$ to 60, which provides the sparsity we want. 

```@example sim
using HighDimMixedModels
out = hdmm(X, G, y, gr; λ = 60)
```

Similary, we can fit a model with the LASSO penalty:

```@example sim
out_las = hdmm(X, G, y, gr; λ = 60, penalty = "lasso")
```

We can compare the estimation performance of the LASSO and SCAD by printing their estimates side by side with the true non-zero parameters values:
```@example sim
[β[1:10] out[1:10] out_las[1:10]]
```

## Real data

### Load dataset
The [cognitive dataset](../data/cognitive.csv) contains data from a [study](https://www.sciencedirect.com/science/article/pii/S0022316623025622) of the effect of an intervention in school lunches among schools in Kenya, accessed via the R package [`splmm`](https://cran.r-project.org/web/packages/splmm/index.html). The data is longitudinal with measurements of students' performance on various tests taken at different points in time. We will fit a model with random intercepts and random growth slopes for each student. Note that while this is a low-dimensional example (``p < n``), the algorithm that this package implements was designed and tested with the high dimensional use-case (``p > n``) in mind.

First, we load the data into Julia and form a categorical variable for the treatment in the study, which was the type of lunch served (assigned at the school level).
```@example cog
using CSV
using DataFrames
using CategoricalArrays
cog_df = CSV.read("../data/cognitive.csv", DataFrame)
# form categorical variable for treatment
cog_df.treatment = categorical(cog_df.treatment, levels=["control", "calorie", "meat", "milk"])
nothing # hide
```

### Extract model matrices, cluster ids, and response vector

Next we form model matrices with the help of the [`StatsModels`](https://juliastats.org/StatsModels.jl/stable/formula/#The-@formula-language) formula syntax:
```@example cog
using StatsModels
f = @formula(ravens ~ 1 + year + treatment + sex + age_at_time0 +
                      height + weight + head_circ + ses + mom_read + mom_write + mom_edu)
model_frame = ModelFrame(f, cog_df)
model_mat = ModelMatrix(model_frame).m
nothing # hide
```

We form two model matrices. One is low dimensional and includes features we do not wish to penalize, and the other is higher dimensional and includes the many features whose effects will be regularized. 
```@example cog
X = model_mat[:, 1:2] # Non-penalized, random effect columns (one for intercept, and the other for year)
G = model_mat[:, 3:end] # High dimensional set of covariates whose effects are regularized
nothing # hide
```
Finally, we get the cluster (in this case, student) ids and the response, the students' Ravens test scores. 

```@example cog
student_id = cog_df.id
y = cog_df.ravens
nothing # hide
```


### Fitting model
The main function in the package is `hdmm()`, which accepts the two design matrices, the response, and the group id as required positional arguments, and returns the fitted model:
```@example cog
using HighDimMixedModels
fit = hdmm(X, G, y, student_id)
```

The function has a number of keyword arguments that can be specified to modify the defaults--see the [documentation](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/lib/public_methods/#HighDimMixedModels.hdmm). For example, by default, we fit a model with the SCAD penalty, which produces less bias. To apply the LASSO penalty, specify `penalty = "lasso"` in the call. Also note that the default value of $\lambda$ (the hyperparameter that controls the degree of penalization) is 10. Since the default (unless `standardize = false`) is to standardize the design matrices before running the algorithm, this is how much the coefficients of the standardized predictors are penalized. In practice, you can and should try fitting the model for several different choices of $\lambda$ and choose the fit that leads to the lowest `fit.BIC`.

By default, the features that are assigned random slopes are all those that appear as columns in the matrix `X`, i.e. those features whose coefficients are not penalized. If you wish to include a feature whose coefficient is not penalized, but do not wish to assign this feature a random slope, then you can specify the argument `Z` in the call to `hdmm` to be a matrix whose columns contain only the variables in `X` that you wish to assign random slopes.

### Inspecting model

The object `fit` which is returned by `hdmm()` is a struct with fields providing all relevant information about the model fit. These can be accessed using the `dot` notation, e.g. `fit.fixef` to retrieve all the fixed effect estimates (including those set to 0) and `fit.log_like` to get the log likelihood at the estimates. To print all the fields stored in the object, you can type `fit.` followed by the tab key or check the [documentation for the struct](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/lib/public_methods/#HighDimMixedModels.HDMModel).

Several model inspection functions from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/tree/master) are also available, such as `residuals(fit)` and `fitted(fit)`. Note that these fitted values and residuals take into account the random effects by incorporating the best prediction of these random effects (BLUPs) for each student into the predictions. 

To print a table with only the selected coefficients (i.e. those that are not set to 0), use the function `coeftable()`. The names of these variables will appear alongside their estimates if you pass them as a second argument:
```@example cog
coeftable(fit, coefnames(model_frame))
```

### Plotting model

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
