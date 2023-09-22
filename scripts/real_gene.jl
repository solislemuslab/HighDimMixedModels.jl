using HighDimMixedModels
using CSV
using DataFrames
using Lasso
using MLBase
using Random
using Serialization

## get data
# data set
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

# random intercept model
X = ones(N, 1)
control = Control()
control.trace = 3
λ = 45
control.tol = 1e-3
Random.seed!(1234)
int_fit = lmmlasso(X, gene_expressions, y, grp; 
    penalty="scad", λ=λ, ψstr="ident", control=control)

# Simple LASSO with no random effect
# Random.seed!(1234)
# lasso_fit = fit(LassoModel, gene_expressions, y; 
#     select = MinCVmse(Kfold(N, 10)))
# beta = coef(lasso_fit)
# println("(# non-zeros, BIC) in LASSO fit: ($(sum(beta .!= 0)), $(round(bic(lasso_fit), digits = 2)))")
# # cycle through each non-zero coefficient estimated from the simple Lasso fit and
# # for each such coefficient, associate a random effect to the corresponding predictor
# # then fit the model with the random effect
# idxs = findall(beta[2:end] .!= 0) #skip the intercept, which we've already done
# gene_names = names(ribo)[3:end][idxs]
# res = Vector{Any}(undef, length(idxs))
# for (i, idx) in enumerate(idxs)[1:3] #skip the intercept, which we've already done
#     println("idx = $idx")
#     # add random effect for the idx-th predictor
#     Z = gene_expressions[:, [idx]]
#     G = gene_expressions[:, Not(idx)]
#     local X = [ones(N) Z]
#     # fit model
#     local control = Control()
#     control.trace = 3
#     control.tol = 1e-3
#     local λ = 45
#     Random.seed!(1234)
#     try 
#         fit = lmmlasso(X, G, y, grp, Z; penalty="scad", λ=λ, ψstr="ident", control=control)
#         println("Model converged! Hurrah!")
#         println("Initial number of non-zeros is $(fit.init_nz)")
#         println("Final number of non-zeros is $(fit.nz)")
#         println("Log likelihood is $(fit.log_like)")
#         println("BIC is $(fit.bic)")
#         println("Estimated L is $(fit.L)")
#         println("Estimated σ² is $(fit.σ²)")
#         res[i] = Base.structdiff(fit, (data=1, weights=1, fitted=1))
#     catch e
#         println("An error occurred in index $(idx): $e")
#         res[i] = "error"
#     end
# end

# We've run the commented out part on the server and can now load results
# Note when we ran on the server, we get only 31 non-zero coefficients in the original
# LASSO fit, versus when we run locally we get in the 40's. This is due to some difference
# between the random number generators on the server (using Julia 1.8) and locally.
res = load_object("data/real/gene_expressions/active_gene_results.jld2")
agn = load_object("data/real/gene_expressions/active_gene_names.jld2")
agn = agn[res .!= "error"]
res = res[res .!= "error"]
ψs = [x.ψ[1] for x in res]
kappa = 0.03
lre_mask = ψs .> kappa
println("Number of genes with high random effect: $(sum(lre_mask))")
println("Gene names with high random effect: $(agn[lre_mask])")
println("ψs for genes with high random effect: $(ψs[lre_mask])")
println("BICs for genes with high random effects $([x.bic for x in res[lre_mask]])")

# fit model with random effect for these two genes
rand_idx = indexin(agn[lre_mask], names(ribo)[3:end])

X =  hcat(ones(N), gene_expressions[:, rand_idx])
Z = gene_expressions[:, rand_idx]
G = gene_expressions[:, Not(rand_idx)]
λ = 50
Random.seed!(1234)
final_fit = lmmlasso(X, G, y, grp, Z; penalty="scad", λ=λ, ψstr="diag", control=control)
final_fit.nz
final_fit.ψ
final_fit.σ²
final_fit.bic
all_gene_names = names(ribo)[3:end];
idx_final = final_fit.fixef[2:end] .!= 0;
final_selected_genes = all_gene_names[idx_final]
save_object("data/real/gene_expressions/final_fit.jld2", final_fit)