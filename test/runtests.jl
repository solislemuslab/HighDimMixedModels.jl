using Revise
import HighDimMixedModels as hdmm
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

include("../scripts/sim_helpers.jl")
import Main.simulations as sim
##Include code for lmmSCAD functions
R"source(\"../R/lmmSCAD/helpers.R\")"
R"library(splmm)"
R"library(emulator)"
R"library(glmnet)"

#Create control struct containing hyper-parameters for algorithm
#For Julia
control = hdmm.Control() 
#For lmmSCAD
R"tol=10^(-2); trace=1; maxIter=1000; maxArmijo=20; number=5; a_init=1; delta=0.1; rho=0.001; gamma=0; 
  lower=10^(-6); upper=10^8;"
#For lmmlmasso
R"control = splmmControl()"

# Simulate a dataset 
p = 5
q = 3
m = 3
X, G, Z, grp = sim.simulate_design(p=p, q=q, m=m)
g = length(unique(grp))
N = length(grp)

XG = [X G]

R"m = $m"
R"p = $p"
R"q = $q"
R"g = $g"
R"N = $N"
R"X = $X"
R"G = $G"
R"Z = $Z"
R"grp = factor(as.numeric($grp))"
R"XG = cbind(X,G)"
R"ll1 <- 1/2*N*log(2*pi)"

# True parameters 
βun = [20, 2, -3]
βpen = [4, -3, 0, 1, 1]
β = [βun; βpen]
L = LowerTriangular([15 0 0; 0 10 0; 0 0 5])
σ² = 100
R"beta = $β"
R"L = $L"
R"sigma2 = $σ²"

# Simulate y
yfixed = X*βun + G*βpen
y = sim.simulate_y(X, G, Z, grp, βun, βpen, L, σ²)
R"yfixed = $yfixed"
R"y = $y"

#Create penalty hyperparameters
λ = .01
a = 3.7
wts = fill(1, p)
λwtd =[zeros(q); λ./wts]
R"λ = $λ"
R"λwtd = $λwtd"
R"a = $a"
R"wts = $wts"


# Estimate fixed effect parameters with LASSO that doesn't take into account random effect structure (with several different settings)
pf = [zeros(q-1); ones(p)]
Random.seed!(54)
lassomod = coef(fit(LassoModel, [X G][:,Not(1)], y; select = MinCVmse(Kfold(size(X)[1], 10)), penalty_factor=pf))
lassomod_nopen = coef(fit(LassoModel, [X G][:,Not(1)], y; select = MinCVmse(Kfold(size(X)[1], 10))))
lassomod_yfixed = coef(fit(LassoModel, [X G][:,Not(1)], yfixed; select = MinCVmse(Kfold(size(X)[1], 10)), penalty_factor=pf))
lassomod_yfixed_nopen = coef(fit(LassoModel, [X G][:,Not(1)], yfixed; select = MinCVmse(Kfold(size(X)[1], 10))))

R"pf = $pf"
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
invVgrp = Vector{Matrix}(undef, g)
hdmm.invV!(invVgrp, Zgrp, L, σ²)
R"XGgrp = $XGgrp"
R"ygrp = $ygrp"
R"Zgrp = $Zgrp"
R"invVgrp = $invVgrp"

## Tests
@testset "log likelihood function" begin
    
    true_ll = hdmm.get_negll(invVgrp, ygrp, XGgrp, β)
    lassomod_ll = hdmm.get_negll(invVgrp, ygrp, XGgrp, lassomod)
    lassomod_nopen_ll = hdmm.get_negll(invVgrp, ygrp, XGgrp, lassomod_nopen)
    lassomod_yfixed_ll = hdmm.get_negll(invVgrp, ygrp, XGgrp, lassomod_yfixed)
    lassomod_yfixed_nopen_ll = hdmm.get_negll(invVgrp, ygrp, XGgrp, lassomod_yfixed_nopen)
    my_log_likes = [true_ll, lassomod_ll, lassomod_nopen_ll, lassomod_yfixed_ll, lassomod_yfixed_nopen_ll]

    R"true_ll = -splmm:::MLloglik(XGgrp, ygrp, invVgrp, beta, N, g, 0)"
    R"lassomod_ll = -splmm:::MLloglik(XGgrp, ygrp, invVgrp, lassomod, N, g, 0)"
    R"lassomod_nopen_ll = -splmm:::MLloglik(XGgrp, ygrp, invVgrp, lassomod_nopen, N, g, 0)"
    R"lassomod_yfixed_ll = -splmm:::MLloglik(XGgrp, ygrp, invVgrp, lassomod_yfixed, N, g, 0)"
    R"lassomod_yfixed_nopen_ll = -splmm:::MLloglik(XGgrp, ygrp, invVgrp, lassomod_yfixed_nopen, N, g, 0)"
    R"splmm_loglikes = c(true_ll, lassomod_ll, lassomod_nopen_ll, lassomod_yfixed_ll, lassomod_yfixed_nopen_ll)"
    @rget splmm_loglikes
    
    @test isapprox(my_log_likes, splmm_loglikes)

end

## Test scad penalty 
@testset "scad value" begin

    scad = hdmm.get_scad(β[(p+1):end], λwtd, a)
    R"Rscad = OSCAD(beta[(p+1):(p+q)], λwtd, a)" 

    scad2 = hdmm.get_scad(lassomod[(p+1):end], λwtd, a)
    R"Rscad2 = OSCAD(lassomod[(p+1):(p+q)], λwtd, a)" 

    @rget Rscad Rscad2
    @test isapprox([scad, scad2], [Rscad, Rscad2])
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
    @test all(isapprox.(my_pars, splmm_pars, atol = 1e-2))

end

### Test fixed effect parameter update functions

#Choose one of the initialized parameters for further testing
βiter = copy(lassomod)
cov_iter = hdmm.cov_start(XGgrp, ygrp, Zgrp, βiter)
Liter = cov_iter[1] #Assume identity structure for now
σ²iter = cov_iter[2]
hdmm.invV!(invVgrp, Zgrp, L, σ²)
R"βiter_R = $βiter"
R"Liter_R = $Liter"
R"σ2iter_R = $σ²iter"
R"invVgrp = $invVgrp"
R"ll2 <- sum(mapply(nlogdetfun,invVgrp))"

#Empty arrays for storing
hess = zeros(p+q)
mat = zeros(g, p+q)
R"hess = rep(0, p+q)"
R"mat = $mat"

active_set = findall(βiter .!= 0)
R"active_set = $active_set"


hdmm.hessian_diag!(XGgrp, invVgrp, active_set, hess, mat[:, active_set])
hess_untrunc = copy(hess)
hess[active_set] = min.(max.(hess[active_set], control.lower), control.upper)

R"hess_R = splmm:::HessianMatrix(XGgrp,invVgrp,active_set,g,hess,mat[,active_set])"
R"hess_untrunc_R = hess_R"
R"hess_R[active_set] = pmin(pmax(hess_R[active_set],control$lower),control$upper)"


R"LxGrp = splmm:::as1(XGgrp, invVgrp, active_set, g)" 

@testset "hessian and cut calculation" begin
    
    @rget hess_untrunc_R
    @test isapprox(hess_untrunc_R, hess_untrunc)
    for j in active_set
        cut = hdmm.special_quad(XGgrp, invVgrp, ygrp, βiter, j)
        R"j = $j"
        R"cut_R = splmm:::as2(XG, y, βiter_R, j, active_set, grp, LxGrp)"
        @rget cut_R
        @test isapprox(cut, cut_R, atol = 1e-6)
    end

end


ll_old = hdmm.get_negll(invVgrp, ygrp, XGgrp, βiter)
fct_old = hdmm.get_cost(ll_old, βiter[(q+1):end], "scad", λwtd[(q+1):end])

R"cut = rep(0, p+q)"
println("Current value of  βiter is $(βiter)")
println("Current function value is $(fct_old)")
for j in active_set
    cut = hdmm.special_quad(XGgrp, invVgrp, ygrp, βiter, j)
    R"cut[$j] = $cut"
    arm = hdmm.armijo!(XGgrp, ygrp, invVgrp, βiter, 
    j, q, cut, hess_untrunc[j], hess[j], 
    "scad", λwtd, a, fct_old, 0, control)
    println("The new value of β is $(βiter) with function value $(arm.fct)")
    global fct_old = arm.fct   
    if arm.arm_con == 1
        @warn "Armijo did not converge"
    end
end

#splmm's armijo doesn't work so we use lmmSCAD's to check
R"cat(\"Current value of βiter is \", βiter_R, \"\n\")"
R" 
for (j in active_set) {
    JinNonpen = j %in% 1:q
    arm = ArmijoRuleSCAD(XGgrp,ygrp,invVgrp, βiter_R, j=j, cut[j], hess_untrunc_R[j], hess_R[j], 
    JinNonpen = JinNonpen, λ, a, weights = rep(1,p+q), 
    nonpen = 1:q , ll1, ll2, converged = 0, 
    control=list(max.armijo=maxArmijo,a_init=a_init,delta=delta,rho=rho,gamma=gamma))
    print(arm)
    βiter_R = arm$b
}
"

@testset "Fixed effect updates" begin
    
    @rget βiter_R
    @test isapprox(βiter_R, βiter, atol = 1e-5)


end

R"βiter = βiter_R"

#We can't test armijo rule for lasso unfortunately


### Test covariance parameter update functions

#Start with testing for identity structure
Lnew = hdmm.L_ident_update(XGgrp, ygrp, Zgrp, βiter, σ²iter, control.var_int, control.thres)


R"activeset <- which(βiter!=0)"
R"rIter <- y-XG[,activeset,drop=FALSE]%*%βiter[activeset,drop=FALSE]"
R"resGrp <- split(rIter,grp)"

R"optRes <- nlminb(Liter_R,MLpdIdentFct,zGroup=Zgrp,resGroup=resGrp, sigma=sqrt(σ2iter_R),
LPsi=diag(q),lower = 10^(-6), upper = 100)"


@testset "L identity structure update function" begin

    @rget optRes 
    @test isapprox(optRes[:par], Lnew, atol = 1e-4)

end

##Now test diagonal structure
Ldiag = fill(Liter, m)
R"Ldiag_R = rep(Liter_R, m)"
#First component
s = 1
hdmm.L_diag_update!(Ldiag, XGgrp, ygrp, Zgrp, βiter, σ²iter, s, control.var_int, control.thres)
Ldiag
R"s = $s"
R"optRes <- nlminb(Ldiag_R[s],MLpdSymFct,zGroup=Zgrp,resGroup=resGrp, sigma=sqrt(σ2iter_R),
                         a=s,b=s,LPsi=diag(Ldiag_R, nrow=length(Ldiag_R)),lower = 10^(-6), upper = 100)"
R"Ldiag_R[s] = optRes$par"
#Second component
s = 2
hdmm.L_diag_update!(Ldiag, XGgrp, ygrp, Zgrp, βiter, σ²iter, s, control.var_int, control.thres)
Ldiag
R"s = $s"
R"optRes <- nlminb(Ldiag_R[s],MLpdSymFct,zGroup=Zgrp,resGroup=resGrp, sigma=sqrt(σ2iter_R),
                         a=s,b=s,LPsi=diag(Ldiag_R, nrow=length(Ldiag_R)),lower = 0, upper = 100)"
R"Ldiag_R[s] = optRes$par"

s = 3
hdmm.L_diag_update!(Ldiag, XGgrp, ygrp, Zgrp, βiter, σ²iter, s, control.var_int, control.thres)
Ldiag
R"s = $s"
R"optRes <- nlminb(Ldiag_R[s],MLpdSymFct,zGroup=Zgrp,resGroup=resGrp, sigma=sqrt(σ2iter_R),
                         a=s,b=s,LPsi=diag(Ldiag_R, nrow=length(Ldiag_R)),lower = 0, upper = 100)"
R"Ldiag_R[s] = optRes$par"


@testset "L diagonal structure update function" begin

    @rget Ldiag_R 
    @test isapprox(Ldiag, Ldiag_R, atol = 1e-4)

end


##Now test general symmetric structure

#Create general lower traingular L (assuming diagoanl entries have already been updated)
Lsym = LowerTriangular(diagm(Ldiag)) 
#Update off diagonal entries 
for i in 2:m
    for j in 1:(i-1)
        hdmm.L_sym_update!(Lsym, XGgrp, ygrp, Zgrp, βiter, σ²iter, (i,j), 
        control.var_int, control.cov_int, control.thres)
    end
end

R"Lsym_R = diag(Ldiag_R, nrow=length(Ldiag_R))" 
R"for (i in 2:m) {
    for (j in 1:(i-1)) {
        optRes <- nlminb(0,MLpdSymFct,zGroup=Zgrp,resGroup=resGrp, sigma=sqrt(σ2iter_R), 
        a=i,b=j,LPsi=Lsym_R,lower = -500, upper = 500)
        Lsym_R[i, j] = optRes$par
    }
}"


@testset "L symmetric structure update function" begin

    @rget Lsym_R 
    #print(Lsym - Lsym_R)
    @test isapprox(Lsym, Lsym_R, atol = 1e-4)

end

R"PsiIter <- Lsym_R %*% t(Lsym_R)" 

## Finally, test sigma update
σ²iter = hdmm.σ²update(XGgrp, ygrp, Zgrp, βiter, Lsym, control.var_int)
R"optRes <- nlminb(sqrt(σ2iter_R),MLsigmaFct,zGroup=Zgrp,resGroup=resGrp,Psi=PsiIter,lower = 0, upper = 100)"
R"σ2iter_R = optRes$par^2"

@testset "error variance update" begin

    @rget σ2iter_R 
    #print(Lsym - Lsym_R)
    @test isapprox(σ²iter, σ2iter_R , atol = 1e-3)

end



























