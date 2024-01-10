using HighDimMixedModels
using CSV
using DataFrames
using Lasso
using MLBase
using Random
using JLD2

# get data
data_dir = "data/real/OTU"
d = CSV.read("$data_dir/tss_normalized_data.csv", DataFrame)


N = size(d)[1]
grp = d.country
otus = Matrix(d[:,2:(end-1)])
y = d.age
X = ones(N, 1)

control = Control()
control.trace = 3
λ = 75
control.tol = 1e-3
Random.seed!(1234)
# We choose to standardize because the data has been library sum scaled and so is very small numbers
int_fit = lmmlasso(X, otus, y, grp; 
    penalty="scad", λ=λ, ψstr="ident", control=control)

save_object("data/real/OTU/otu_fit.jld2", int_fit)


# load otu_fit.jld2
otu_fit = load_object("data/real/OTU/otu_fit.jld2")
