using Revise
using HighDimMixedModels
using Random
using RCall
include("simulations.jl")
using .simulations
R"library(splmm)"



######## Comparison results of a single instance to splmm ###########

#Get design matrix
g = 25; n = fill(6, g); p = 1; q = 1000; m = p; rho = 0.2;
Random.seed!(350)
X, G, Z, grp = simulate_design(; n=n, p=p, q=q, m=m, rho=rho)


#Specify parameters
βun = [1]
βpen = vcat([2,4,3,3], zeros(q-4))
theta = Lid = sqrt(0.56)
σ² = 0.25^2
y = simulations.simulate_y(X, G, Z, grp, βun, βpen, theta, σ²)
control = Control()

est = lmmlasso(X, G, y, grp, Z; 
        standardize = false, penalty = "scad", 
        λ=150.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
        init_coef = nothing, ψstr="ident", control=control)




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










Random.seed!(350)
function simulation()    
    d = Normal()
    x = rand(d, 5)
end


function simple_simulations()
    for _ in 1:10
        y = simulations.simulate_y(X, G, Z, grp, βun, βpen, Lid, σ²)
        println(y[1:3])
    end
end

simple_simulations()




    
    


