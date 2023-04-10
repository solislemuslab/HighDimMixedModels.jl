using Revise
using HighDimMixedModels
using Random
using RCall
include("sim_helpers.jl")
using Main.simulations #Exports simulate_design() and simulate_y()
R"library(splmm)"



######## Comparison results of a single instance to splmm ###########

#Get design matrix
g = 50; n = fill(3, g); q = 2; p = 1000; m = q; rho = 0.2;
Random.seed!(350)
X, G, Z, grp = simulate_design(; n=n, q=q, p=p, m=m, rho=rho)


#Specify parameters
βun = [1, 2]
βpen = vcat([4,3,3], zeros(q-3))
theta = Lid = sqrt(0.56)
σ² = 0.25^2
y = simulations.simulate_y(X, G, Z, grp, βun, βpen, theta, σ²)
control = Control()

est = lmmlasso(X, G, y, grp, Z; 
        standardize = false, penalty = "scad", 
        λ=100.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
        init_coef = nothing, ψstr="ident", control=control)


R"X = $X"
R"G = $G"
R"XG = cbind(X, G)"
R"y = $y"
R"Z = $Z"
R"grp = factor($grp, levels = 1:$g)"
R"fit = splmm(x=XG,y=y,z=X,grp=grp,lam1=100.0,lam2=10.0, nonpen.L = 1:3, penalty.b=\"scad\", penalty.L=\"scad\")"


####Simulations######

""" 
This function is fed design matrices, grouping info, and parameters. 
It then generates nsim response vectors y and runs the lmmlasso algorithm to obtain an estimate of the parameters for each.
You must also specify the hyper-parameter λ.
"""
function run_simulations(X, G, Z, grp, βun, βpen, L, ψstr, σ², control, nsim, λ)

    βsim = zeros(nsim, length(vcat(βun, βpen)))
    Lsim = zeros(nsim, isa(L, Number) ? 1 : length(L[:]))
    σ²sim = zeros(nsim)
    iter_sim = zeros(nsim)

    for i in 1:nsim
        y = simulations.simulate_y(X, G, Z, grp, βun, βpen, L, σ²)

        println(y[1:3])
        est = lmmlasso(X, G, y, grp, Z; 
        standardize = false, penalty = "scad", 
        λ=λ, scada = 3.7, wts = fill(1.0, size(G)[2]), 
        init_coef = nothing, ψstr=ψstr, control=control)

        βsim[i,:] .= est[1]
        Lsim[i,:] .= est[2]
        σ²sim[i] = est[3]
        iter_sim[i] = est[4]
    
    end

    return (β=βsim, L=Lsim, σ²=σ²sim, iter=iter_sim)

end


NSIM = 100


sim_results = run_simulations(X, G, Z, grp, βun, βpen, Ldiag, "diag", σ², control, NSIM, 25.0)






    
    


