# Example data

## Load dataset
The [cognitive dataset](../data/cognitive.csv) contains data from a [study](https://www.sciencedirect.com/science/article/pii/S0022316623025622) of the effect of an intervention in school lunches among schools in Kenya, accessed via the R package [`splmm`](https://cran.r-project.org/web/packages/splmm/index.html). The data is longitudinal with measurements of students' performance on various tests taken at different points in time. We will fit a model with random intercepts and random growth slopes for each student. Note that while this is a low-dimensional example (``p < n``), the algorithm that this package implements was designed and tested with the high dimensional use-case (``p > n``) in mind.

First, we load the data into Julia and form a categorical variable for the treatment in the study, which was the type of lunch served (assigned at the school level).
```@example cog
using CSV
using DataFrames
using CategoricalArrays
cog_df = CSV.read("../data/cognitive.csv", DataFrame)
# form categorical variable for treatment
cog_df.treatment = categorical(cog_df.treatment, levels=["control", "calorie", "meat", "milk"])
nothing #hide
```

## Extract model matrices, cluster ids, and response vector

Next we form model matrices with the help of the [`StatsModels`](https://juliastats.org/StatsModels.jl/stable/formula/#The-@formula-language) formula syntax:
```@example cog
using StatsModels
f = @formula(ravens ~ 1 + year + treatment + sex + age_at_time0 +
                      height + weight + head_circ + ses + mom_read + mom_write + mom_edu)
model_frame = ModelFrame(f, cog_df)
model_mat = ModelMatrix(model_frame).m
nothing #hide
```

We form two model matrices. One is low dimensional and includes features we do not wish to penalize, and the other is higher dimensional and includes the many features whose effects will be regularized. 
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
```


