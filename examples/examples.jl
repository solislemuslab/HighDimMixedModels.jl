using Revise
using HighDimMixedModels #Package
using Random 
using LinearAlgebra
using Lasso


"""
Simulates a data set

ARGUMENTS
- N :: Number of groups
- nᵢ :: Vector of lenght N with entries giving the number of observations per group 
- p :: Number of fixed effect covariates
- m :: Number of random effect covariates
- β :: Vector of length p+1 (to account for intercept) with fixed effect parameters
- X :: Fixed effect design matrix
- Z :: Random effect design matrix
- L :: Random effect variance component
- σ² :: Error variance

OUTPUT
- Xgrp :: List of length the number of groups, each of whose elements is the fixed effect design matrix for the group
- Zgrp :: List of length the number of groups, each of whose elements is the random effect design matrix for the group
- ygrp :: List of length the number of groups, each of whose elements is the response for the group
"""
function simulate(; N=30, nᵢ=fill(12, N), p=6, m=2, β=[5,10,20,10,0,0,0], 
    X=hcat(fill(1,Ntot), randn(Ntot, p)), Z=X[:, 1:m], L=10, σ²=100)
    
    Ntot = sum(nᵢ) #Total number of observations
    grp = string.(repeat(1:N, inner = nᵢ[1])) #grouping variable
    b = L*randn(N, m)
    
    y = Float64[]
    for (i, group) in enumerate(unique(grp))
        Zᵢ = Z[grp .== group,:]
        Xᵢ = X[grp .== group,:]
        bᵢ = b[i,:]
        yᵢ = Xᵢ*β + Zᵢ*bᵢ + sqrt(σ²)*randn(nᵢ[1])
        y = vcat(y, yᵢ)
    end

    #Grouped data
    Zgrp, Xgrp, ygrp = Matrix[],Matrix[],Vector[]
    for group in unique(grp)
        Zᵢ, Xᵢ, yᵢ = Z[grp .== group,:], X[grp .== group,:], y[grp .== group]
        push!(Zgrp, Zᵢ); push!(Xgrp, Xᵢ); push!(ygrp, yᵢ)
    end
    return Xgrp, Zgrp, ygrp

end

## Lasso fit ignoring the random effects
function lasso(X, y)
    lassopath = fit(LassoPath, X[:,2:end], y)
    β = coef(lassopath; select=MinBIC())
    return β
end



#Simulate a data-set
Random.seed!(540)
N = 30; nᵢ = 80; p=28; m=3  
β=[15; repeat([5,10,20,10,0,0,0], 4)]
X = hcat(fill(1, N*nᵢ), randn(N*nᵢ, p))
L = 100; σ² = 100
Xgrp, Zgrp, ygrp = simulate(; β=β, N=N, nᵢ=nᵢ, X=X, L=L, σ²=σ², p=28, m=m)


## Initial parameters 
β₀ = lasso(X, vcat(ygrp...))
println("Lasso estimated fixed effect are $β₀")
Lest, σ²est = cov_start(Xgrp, ygrp, Zgrp, β)
println("(L, σ²) calculated from true fixed parameters is $((Lest, σ²est))")
println("(L, σ²) calculated from Lasso estimate $(cov_start(Xgrp, ygrp, Zgrp, β₀))")

Lest = fill(Lest, 3)
map(s->L_diag_update!(Lest, Xgrp, ygrp, Zgrp, β₀, σ²est, s), 1:m)
Lest