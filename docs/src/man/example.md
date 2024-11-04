# Examples

We'll first shown an example with simulated data where we know the ground-truth parameters, and then show an example with real data.

## Simulated data

We first generate two design matrices. The first, `X`, will be low dimensional and consist of the predictors that will have associated random effects--this includes a column of constant 1s so that we include random intercepts. The second, `G`, is high dimensional and includes predictors whose coefficients we want to penalize.

We also generate a sparse vector of ground-truth regression coefficient, $\beta$, whose non-zero components are centered at 0 and have a standard deviation of 1. 
```@example sim
using Random
Random.seed!(420)
g, n, p, q = 50, 5, 1000, 4 # #groups, #samples per group, #features, #features with random effect (including random intercept)
N = n*g  # Total sample size
X = [ones(N) randn(N, q-1)]
G = randn(N, p-q)
XG = [X G] 
sd_signal = 1
β = [sd_signal .* randn(10); zeros(p-10)]
nothing # hide
```

Now we generate the response from these design matrices
```@example sim
using Distributions
using LinearAlgebra
# Generate group assignments
gr = string.( vcat( [fill(i, n) for i in 1:g]... ) )

# Generate random effects
ψ = Diagonal(1:4) #random effects variances
dist_b = MvNormal(zeros(q), ψ) 
b = rand(dist_b, g)

# Generate response
y_fixed = XG * β 
y = Vector{Float64}(undef, N)
for (i, group) in enumerate(unique(gr))
    group_ind = (gr .== group)
    nᵢ = sum(group_ind)
    Xᵢ = X[group_ind,:]
    bᵢ = b[:,i]
    yᵢ = Xᵢ*bᵢ + randn(nᵢ)
    y[group_ind] = y_fixed[group_ind] + yᵢ
end
nothing # hide
```

We can now fit the model with the function `hdmm` exported by the package. It accepts the two design matrices, the response, and the group id as required positional arguments, and returns the fitted model. The default is to use the SCAD penalty and we choose penalty severity $\lambda = 25$.

```@example sim
using HighDimMixedModels
out_scad = hdmm(X, G, y, gr; λ = 25)
```

We see $\lambda = 25$ yields a sparse model but without every penalized coefficient set to 0. In practice, we'd want to experiment with the penalty severity $\lambda$ to find the model that minimizes `out_scad.bic`.

Similarly, we can fit a model with the LASSO penalty:

```@example sim
out_las = hdmm(X, G, y, gr; λ = 25, penalty = "lasso")
```

We can compare the estimation performance of the LASSO and SCAD by printing their fixed effect coefficient estimates, saved at `out.fixef` or `out_las.fixef`, side by side with the true non-zero parameters values. Since the initialization of our descent algorithm, saved at `out.init_coef`, is obtained by fitting a (cross-validated) LASSO model that doesn't take the random effects into account, we also display these estimates to see how we improve by accounting for the random effects.

```@example sim
using DataFrames
DataFrame(
    :true_coefs => β[1:10], 
    :lasso_no_random => out_las.init_coef.βstart[1:10],
    :lasso_with_random => out_las.fixef[1:10], 
    :scad_with_random => out_scad.fixef[1:10]
    )
```


## Real data

### Load dataset
The [cognitive dataset](../data/cognitive.csv) contains data from a [study](https://www.sciencedirect.com/science/article/pii/S0022316623025622) of the effect of an intervention in school lunches among schools in Kenya, accessed via the R package [`splmm`](https://cran.r-project.org/web/packages/splmm/index.html). The data is longitudinal with measurements of students' performance on various tests taken at different points in time. We will fit a model with random intercepts and random growth slopes for each student. Note that while this is a low-dimensional example (``p < n``), the algorithm that this package implements was designed and tested with the high dimensional use-case (``p > n``) in mind.

First, we load the data into Julia, remove the two outlier students with Ravens scores (our response variable) below 10, and form a categorical variable for the treatment in the study, which was the type of lunch served (assigned at the school level).
```@example cog
using CSV
using DataFrames
using CategoricalArrays
cog_df = CSV.read("../data/cognitive.csv", DataFrame)
# remove outlier students
outly_mask = cog_df.id .∈ ([310, 392],)
cog_df = cog_df[.!outly_mask, :]
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

We now form three matrices that will be accepted as input by `hdmm()`: `X` is a matrix whose columns will receive non-penalized fixed effects, `G` is a matrix whose columns will receive penalized fixed effects, and `Z` is a matrix whose columns will receive random effects. Almost always, `Z` will be a subset of `X` and its default in `hdmm()` is to equal `X`. In our case, however, we want to leave unpenalized all the treatment dummy variables, as well as the control intercept and time variable, but we only want to assign random effects to the control intercept and the time variable, so we have to specify `Z` explicitly. 

```@example cog
X = model_mat[:, 1:5] # Non-penalized columns (one for intercept, one for year, 3 for treatment categories) 
Z = X[:, 1:2] # Random effect columns (one for intercept, one for year)
G = model_mat[:, 6:end] # Set of covariates whose effects are regularized
nothing # hide
```

Finally, we get the cluster (in this case, student) ids and the response, the students' Ravens test scores. 

```@example cog
student_id = cog_df.id
y = cog_df.ravens
nothing # hide
```

### Fitting model

```@example cog
using HighDimMixedModels
fit = hdmm(X, G, y, student_id, Z)
```

Note that the default value of $\lambda$ (the hyperparameter that controls the degree of penalization) is 10. Since the default (unless `standardize = false`) is also to standardize the penalized design matrix `G` before fitting the model, this is how much the coefficients of the standardized predictors are penalized. In practice, you can and should try fitting the model for several different choices of $\lambda$ and choose the fit that leads to the lowest `fit.bic`.

### Inspecting model

The object `fit` which is returned by `hdmm()` is a struct with fields providing all relevant information about the model fit. These can be accessed using dot syntax, e.g. `fit.fixef` to retrieve all the fixed effect estimates (including those set to 0) and `fit.log_like` to get the log likelihood at the estimates. To print all the fields stored in the object, you can type `fit.` followed by the tab key or check the [documentation for the struct](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/lib/public_methods/#HighDimMixedModels.HDMModel).

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


