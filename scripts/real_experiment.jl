using HighDimMixedModels
using CSV
using DataFrames
using Lasso
using MLBase
using Random

## get data
# data set
data_dir = "data/real/gene_expressions"
ribo = CSV.read("$data_dir/riboflavingrouped.csv", DataFrame; 
    transpose = true, types = Dict(1 => String)) 
gene_expressions = Matrix{Float64}(ribo[:,3:end])
y = ribo[:,2]
N = length(y)
# group info
grp = readlines("$data_dir/riboflavingrouped_structure.csv")
grp = [replace(x, "\"" => "") for x in grp]


# no random effect
Random.seed!(1234)
lasso_fit = fit(LassoModel, gene_expressions, y; 
    select = MinCVmse(Kfold(N, 10)))
beta = coef(lasso_fit)
println("(# non-zeros, BIC) in LASSO fit: ($(sum(beta .!= 0)), $(round(bic(lasso_fit), digits = 2)))")

# random intercept
X = ones(N, 1)
control = Control()
control.trace = 3
λ = 45
control.tol = 1e-3
Random.seed!(1234)
int_fit = lmmlasso(X, gene_expressions, y, grp; 
    penalty="scad", λ=λ, ψstr="ident", control=control)

# cycle through each non-zero coefficient estimated from the original Lasso fit that did not include random effects
# for each such coefficient, associate a random effect to the corresponding predictor
# then fit the model with the random effect
idxs = findall(beta[2:end] .!= 0) #skip the intercept, which we've already done
res = Vector{Any}(undef, length(idxs))
for (i, idx) in enumerate(idxs) #skip the intercept, which we've already done
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
        fit = lmmlasso(X, G, y, grp, Z; penalty="scad", λ=λ, ψstr="ident", control=control)
        println("Model converged! Hurrah!")
        println("Initial number of non-zeros is $(fit.init_nz)")
        println("Final number of non-zeros is $(fit.nz)")
        println("Log likelihood is $(fit.log_like)")
        println("BIC is $(fit.bic)")
        println("Estimated L is $(fit.L)")
        println("Estimated σ² is $(fit.σ²)")
        res[i] = Base.structdiff(fit, (data=1, weights=1, fitted=1))
    catch e
        println("An error occurred in index $(idx): $e")
        res[i] = "error"
    end
end






# lassopath = fit(LassoModel, XG[:, Not(1)], y;
#             penalty_factor=[zeros(q - 1); 1 ./ wts], select=MinCVmse(Kfold(N, 10)))
# βstart = coef(lassopath)

# control = Control()
# control.trace = 3