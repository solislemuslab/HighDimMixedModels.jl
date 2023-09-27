using Revise
using HighDimMixedModels
using RCall
using DataFrames
using StatsBase
using Random
using MLBase
using Lasso


R"""
library(BGLR)
data(mice)
G = mice.X
snp_names = colnames(G)
X = mice.pheno[,c("GENDER", "Biochem.Age")]
#replace missing values in age column with median
X[is.na(X$Biochem.Age), "Biochem.Age"] = median(X$Biochem.Age, na.rm = TRUE)
y = mice.pheno$Obesity.BMI
grp = mice.pheno$cage
"""

@rget G
@rget snp_names
@rget X
# check that no columns missing values
sum(ismissing.(X.Biochem_Age))
sum(ismissing.(X.GENDER))
@rget y
N = length(y)
@rget grp
grp = String.(grp)

X.ngender = ifelse.(X.GENDER .== "F", 0, 1)
Xmat = hcat(ones(N), Matrix(X[:,["Biochem_Age", "ngender"]]))
Z = ones(N,1)
q = 3
p = size(G, 2)
XG = [Xmat G]
Random.seed!(1234)
lassopath = fit(LassoModel, XG[:, Not(1)], y; maxncoef = 1_000_000,
penalty_factor=[zeros(q-1); fill(1, p)], select=MinCVmse(Kfold(N, 10)));
println("""
Number of non-zero coefs in initial LASSO fit: $(sum(coef(lassopath) .!= 0))
""")

Xmat_small = Xmat[1:500, :]; G_small = G[1:500, :]; 
y_small = y[1:500]; grp_small = grp[1:500]; Z_small = Z[1:500, :];
λ = 2000
control = Control()
control.trace = 3
control.tol = 1e-3
Random.seed!(1234)
gwas_fit2 = lmmlasso(Xmat, G, y, grp, Z; standardize = false,
    penalty="scad", λ=λ, ψstr="ident", control=control)
snp_names[gwas_fit2.fixef[4:end] .!= 0]











