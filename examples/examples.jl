using Revise
using HighDimMixedModels #Package
using Random
using Distributions
using LinearAlgebra
using Lasso
using StatsBase

"""
Simulates a data set

ARGUMENTS
- grp :: Vector of strings of same length as number of rows of X, assigning each observation to a particular group 
- β :: Vector of length p+1 (to account for intercept) with fixed effect parameters
- X :: Fixed effect design matrix
- Z :: Random effect design matrix
- L :: Scalar, Vector, or Lower Triangular matrix depending on how random effect covariance structure is parameterized
- σ² :: Error variance

OUTPUT
- y :: Vector of responses
"""
function simulate(grp, β, X, Z, L, σ²)
    
    groups = unique(grp)
    N = length(groups)
    m = size(Z)[2] #Number of random effects
    Ntot = size(X)[1] #Total Number of observations
    
    ndims(L) < 2 || (L = Matrix(L)*Matrix(L')) #If L is a matrix, it should be a lower triangular matrix
    dist_b = MvNormal(zeros(m), L) 
    b = rand(dist_b, N) #Matrix of dimensions m by N
    y = Vector{Float64}(undef, Ntot)

    for (i, group) in enumerate(groups)
        group_ind = (grp .== group)
        nᵢ = sum(group_ind)
        Zᵢ = Z[group_ind,:]
        Xᵢ = X[group_ind,:]
        bᵢ = b[:,i]
        yᵢ = Xᵢ*β + Zᵢ*bᵢ + sqrt(σ²)*randn(nᵢ)
        y[group_ind] = yᵢ
    end

    return y

end

## Lasso fit ignoring the random effects
function lasso(X, y)
    lassopath = fit(LassoPath, X[:,2:end], y)
    β = coef(lassopath; select=MinCVmse(lassopath, 10))
    return β
end


#Simulate the data as done here: https://rdrr.io/cran/lmmlasso/man/lmmlasso.html
Random.seed!(54)
N = 20 #Number of groups
p = 6 #Number of covariates not including intercept
q = 2 #Number of covariates that have associated random effects
n = fill(6, N) #Observations per group
grp = string.(inverse_rle(1:N, n))  #grouping variable


β=[10,20,40,30,0,0,50]
X = hcat(fill(1, sum(n)), randn(sum(n), p))
Z = X[:,1:q]
L = LowerTriangular([30 0; 0 30])
σ² = 10
y = simulate(grp, β, X, Z, L, σ²)
Zgrp, Xgrp, ygrp = Matrix[], Matrix[], Vector[]
for group in unique(grp)
    Zᵢ, Xᵢ, yᵢ = Z[grp .== group,:], X[grp .== group,:], y[grp .== group]
    push!(Zgrp, Zᵢ); push!(Xgrp, Xᵢ); push!(ygrp, yᵢ)
end

## Initial parameters 
β₀ = lasso(X, y)

println("(L, σ²) calculated from true fixed parameters is $(cov_start(Xgrp, ygrp, Zgrp, β))")
println("(L, σ²) calculated from Lasso estimate $(cov_start(Xgrp, ygrp, Zgrp, β₀))")

#Lest = fill(Lest, 3)
#map(s->L_diag_update!(Lest, Xgrp, ygrp, Zgrp, β₀, σ²est, s), 1:m)
#Lest



