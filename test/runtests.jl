using Revise
using Random
using Distributions
using LinearAlgebra
using Lasso
using StatsBase
using Test
using Optim
using InvertedIndices
using Parameters
using RCall
using MLBase

include("../src/HighDimMixedModels.jl3")
include("../src/simulations.jl")
import Main.HighDimMixedModels as hdmm
import Main.simulations as sim

# Simulate a dataset 
p = 1
q = 5
X, G, Z, grp = sim.simulate_design()
g = length(unique(grp))
N = length(grp)

XG = [X G]

R"g = $g"
R"N = $N"
R"X = $X"
R"G = $G"
R"XG = cbind(X,G)"

# True parameters 
βun = [1,2]
βpen = [4,3,0,0,5]
β = [βun; βpen]
L = LowerTriangular([25 0; 0 3])
σ² = 1
R"beta = $β"
R"L = $L"
R"sigma2 = $σ²"

# Simulate y
yfixed = X*βun + G*βpen
y = sim.simulate_y(X, G, Z, grp, βun, βpen, L, σ²)

R"yfixed = $yfixed"
R"y = $y"

# Estimate fixed effect parameters with LASSO that doesn't take into account random effect structure (with several different settings)
pf = [zeros(p); ones(q)]
Random.seed!(54)
lassomod = coef(fit(LassoModel, [X G][:,Not(1)], y; select = MinCVmse(Kfold(size(X)[1], 10)), penalty_factor=pf))
lassomod_nopen = coef(fit(LassoModel, [X G][:,Not(1)], y; select = MinCVmse(Kfold(size(X)[1], 10))))
lassomod_yfixed = coef(fit(LassoModel, [X G][:,Not(1)], yfixed; select = MinCVmse(Kfold(size(X)[1], 10)), penalty_factor=pf))
lassomod_yfixed_nopen = coef(fit(LassoModel, [X G][:,Not(1)], yfixed; select = MinCVmse(Kfold(size(X)[1], 10))))

R"library(glmnet)"
R"pf = $pf"
R"pf = c(pf)"
R"cvfit <- cv.glmnet(XG[,-1], y, penalty.factor = pf)" # Defaults to 10-fold cv
R"lassomod = coef(cvfit, s = \"lambda.min\")" 
R"cvfit_nopen <- cv.glmnet(XG[,-1], y)" # Defaults to 10-fold cv
R"lassomod_nopen = coef(cvfit_nopen, s = \"lambda.min\")" 
R"cvfit_yfixed = cv.glmnet(XG[,-1], yfixed, penalty.factor = pf)"
R"lassomod_yfixed = coef(cvfit_yfixed, s = \"lambda.min\")"
R"cvfit_yfixed_nopen <- cv.glmnet(XG[,-1], yfixed)" # Defaults to 10-fold cv
R"lassomod_yfixed_nopen = coef(cvfit_yfixed_nopen, s = \"lambda.min\")" 

# Overwrite glmnet's solutions with Julia's solutions for future tests
R"lassomod = $lassomod"
R"lassomod_nopen = $lassomod_nopen"
R"lassomod_yfixed = $lassomod_yfixed"
R"lassomod_yfixed_nopen = $lassomod_yfixed_nopen"

# Get grouped data
XGgrp, Zgrp, ygrp = Matrix[], Matrix[], Vector[]
for group in unique(grp)
    XGᵢ, Zᵢ, yᵢ = XG[grp .== group,:], Z[grp .== group,:], y[grp .== group]
    push!(XGgrp, XGᵢ); push!(Zgrp, Zᵢ); push!(ygrp, yᵢ)
end
Vgrp = hdmm.Vgrp(L, Zgrp, σ²)
Lgrp = [inv(Symmetric(V)) for V in Vgrp]
R"XGgrp = $XGgrp"
R"ygrp = $ygrp"
R"Zgrp = $Zgrp"
R"Lgrp = $Lgrp"



## Tests
@testset "log likelihood function" begin
    
    true_ll = hdmm.negloglike(Vgrp, ygrp, XGgrp, β)
    lassomod_ll = hdmm.negloglike(Vgrp, ygrp, XGgrp, lassomod)
    lassomod_nopen_ll = hdmm.negloglike(Vgrp, ygrp, XGgrp, lassomod_nopen)
    lassomod_yfixed_ll = hdmm.negloglike(Vgrp, ygrp, XGgrp, lassomod_yfixed)
    lassomod_yfixed_nopen_ll = hdmm.negloglike(Vgrp, ygrp, XGgrp, lassomod_yfixed_nopen)
    my_log_likes = [true_ll, lassomod_ll, lassomod_nopen_ll, lassomod_yfixed_ll, lassomod_yfixed_nopen_ll]

    R"library(splmm)"
    R"true_ll = -splmm:::MLloglik(XGgrp, ygrp, Lgrp, beta, N, g, 0)"
    R"lassomod_ll = -splmm:::MLloglik(XGgrp, ygrp, Lgrp, lassomod, N, g, 0)"
    R"lassomod_nopen_ll = -splmm:::MLloglik(XGgrp, ygrp, Lgrp, lassomod_nopen, N, g, 0)"
    R"lassomod_yfixed_ll = -splmm:::MLloglik(XGgrp, ygrp, Lgrp, lassomod_yfixed, N, g, 0)"
    R"lassomod_yfixed_nopen_ll = -splmm:::MLloglik(XGgrp, ygrp, Lgrp, lassomod_yfixed_nopen, N, g, 0)"
    R"splmm_loglikes = c(true_ll, lassomod_ll, lassomod_nopen_ll, lassomod_yfixed_ll, lassomod_yfixed_nopen_ll)"
    @rget splmm_loglikes
    
    @test isapprox(my_log_likes, splmm_loglikes)

end

@testset "starting parameters" begin

    start1 = hdmm.cov_start(XGgrp, ygrp, Zgrp, β)
    start2 = hdmm.cov_start(XGgrp, ygrp, Zgrp, lassomod)
    start3 = hdmm.cov_start(XGgrp, ygrp, Zgrp, lassomod_nopen)
    start4 = hdmm.cov_start(XGgrp, ygrp, Zgrp, lassomod_yfixed)
    start5 = hdmm.cov_start(XGgrp, ygrp, Zgrp, lassomod_yfixed_nopen)
    my_pars = [collect(start1), collect(start2), collect(start3), collect(start4), collect(start5)]

    R"IdGroup = lapply(Zgrp, function(z) diag(dim(z)[[1]]))"
    R"start1 = splmm:::covStartingValues(XGgrp, ygrp, Zgrp, IdGroup, beta, N, g)"
    R"start2 = splmm:::covStartingValues(XGgrp, ygrp, Zgrp, IdGroup, lassomod, N, g)"
    R"start3 = splmm:::covStartingValues(XGgrp, ygrp, Zgrp, IdGroup, lassomod_nopen, N, g)"
    R"start4 = splmm:::covStartingValues(XGgrp, ygrp, Zgrp, IdGroup, lassomod_yfixed, N, g)"
    R"start5 = splmm:::covStartingValues(XGgrp, ygrp, Zgrp, IdGroup, lassomod_yfixed_nopen, N, g)"
    R"splmm_pars = lapply(list(start1, start2, start3, start4, start5), function(start) c(start$tau, start$sigma^2))"
    @rget splmm_pars

    @test all(isapprox.(my_pars, splmm_pars, atol = 1e-3))

end











