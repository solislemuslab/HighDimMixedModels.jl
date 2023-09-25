using HighDimMixedModels
using CSV
using DataFrames
using Lasso
using MLBase
using Random
using Serialization
using JLD2

# get data
data_dir = "data/real/gene_expressions"
ribo = CSV.read("$data_dir/riboflavingrouped.csv", DataFrame; 
    transpose = true, types = Dict(1 => String)) 
gene_expressions = Matrix{Float64}(ribo[:,3:end])
gene_names = names(ribo)[3:end]
y = ribo[:,2]
N = length(y)
# group info
grp = readlines("$data_dir/riboflavingrouped_structure.csv")
grp = [replace(x, "\"" => "") for x in grp]

# Fit random intercept model 
# and save the names of the genes with non-zero effcts in this model
# Note that the model results in no variation in intercepts 
# (so effectively the same as a LASSO wtihout random effects)
X = ones(N, 1)
control = Control()
control.trace = 3
λ = 45
control.tol = 1e-3
Random.seed!(1234)
int_fit = lmmlasso(X, gene_expressions, y, grp; 
    penalty="scad", λ=λ, ψstr="ident", control=control)
beta = int_fit.fixef
println("Number of non-zero coefs in initial fit: $(int_fit.nz), bic is $(int_fit.bic)")
idxs = findall(beta[2:end] .!= 0) #skip the intercept
gene_names = names(ribo)[3:end][idxs]
save_object("data/real/gene_expressions/gene_names.jld2", gene_names)


# cycle through each non-zero coefficient estimated from the random intercept fit (which resulted in an intercept variance of 0)
# for each such coefficient, associate a random effect to the corresponding predictor
# then fit the model with this random effect
res = Vector{Any}(undef, length(idxs))
for (i, idx) in enumerate(idxs)
    println("idx = $idx")
    # add random effect for the idx-th predictor
    Z = gene_expressions[:, [idx]]
    G = gene_expressions[:, Not(idx)]
    local X = [ones(N) Z]
    # fit model
    local control = Control()
    control.trace = 3
    control.tol = 1e-3
    local λ = 45
    Random.seed!(1234)
    try 
        est = lmmlasso(X, G, y, grp, Z; penalty="scad", λ=λ, ψstr="ident", control=control)
        println("Model converged! Hurrah!")
        println("Initial number of non-zeros is $(est.init_nz)")
        println("Final number of non-zeros is $(est.nz)")
        println("Log likelihood is $(est.log_like)")
        println("BIC is $(est.bic)")
        println("Estimated L is $(est.L)")
        println("Estimated σ² is $(est.σ²)")
        res[i] = Base.structdiff(est, (data=1, weights=1, fitted=1))
    catch e
        println("An error occurred in index $(idx): $e")
        res[i] = "error"
    end
end
save_object("data/real/gene_expressions/gene_results.jld2", res)


###################################################################
# Inspect results and fit a final model with random effects 
# for the genes with the highest random effects from the previous step
res = load_object("data/real/gene_expressions/gene_results.jld2")
gene_names = load_object("data/real/gene_expressions/gene_names.jld2")
#filter out model that didn't converge
gene_names = gene_names[res .!= "error"] 
res = res[res .!= "error"]
# Extract ψs and find genes with high random effects
ψs = [x.ψ[1] for x in res]
kappa = 0.2
lre_mask = ψs .> kappa
println("Number of genes with high random effect: $(sum(lre_mask))")
println("Gene names with high random effect: $(gene_names[lre_mask])")
println("ψs for genes with high random effect: $(ψs[lre_mask])")
rand_idx = indexin(gene_names[lre_mask], names(ribo)[3:end])
X =  hcat(ones(N), gene_expressions[:, rand_idx])
Z = gene_expressions[:, rand_idx]
G = gene_expressions[:, Not(rand_idx)]
λ = 45
Random.seed!(1234)
try 
    global final_fit = lmmlasso(X, G, y, grp, Z; penalty="scad", λ=λ, ψstr="diag", control=control)
    save_object("data/real/gene_expressions/final_fit.jld2", final_fit)
catch e
    println("An error occurred while fitting the final model: $e")
end

# Inspection of final_fit
final_fit = load("data/real/gene_expressions/final_fit.jld2")
print("Final fit: L is $(final_fit.L), σ² is $(final_fit.σ²)")
println("Number of non-zero coefs in final fit: $(final_fit.nz), bic is $(final_fit.bic)")
all_gene_names = names(ribo)[3:end];
idx_final = final_fit.fixef[2:end] .!= 0;
final_selected_genes = all_gene_names[idx_final]
println("Final selected genes: $(final_selected_genes)")