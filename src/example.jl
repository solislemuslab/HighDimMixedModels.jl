using Revise
using HighDimMixedModels
using CSV
using DataFrames
using StatsModels
using CategoricalArrays
using Plots

cog_df = CSV.read("data/cognitive.csv", DataFrame)

# Create model matrices
cog_df.schoolid = categorical(cog_df.schoolid)
cog_df.treatment = categorical(cog_df.treatment, levels=["control", "calorie", "meat", "milk"])
f = @formula(ravens ~ 1 + schoolid + treatment + year + sex + age_at_time0 +
                      height + weight + head_circ + ses + mom_read + mom_write + mom_edu)
mf = ModelFrame(f, cog_df)
mm = ModelMatrix(mf).m
X = mm[:, 1:1] # Intercept
G = mm[:, 2:end]

# Fit the model
grp = cog_df.id
y = cog_df.ravens
control = Control()
control.trace = 3
fit = lmmlasso(X, G, y, grp; control = control)

coeftable(fit, coefnames(mf))

# Plot the fitted values
plot(fit.fitted[map(in(1:2), grp)], y[map(in(1:2), grp)], seriestype=:scatter, c = grp[map(in(1:2), grp)])