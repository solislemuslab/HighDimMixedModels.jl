using HighDimMixedModels
using ZipFile
using CSV
using DataFrames
using DelimitedFiles
using Serialization

data_dir = "data/OTU/random1_covid"
file_name = "rs_data_LinShi_75.csv"

df = CSV.read("$data_dir/$file_name", DataFrame) 
X_names = [col for col in names(df) if startswith(col, "X")]
G_names = [col for col in names(df) if startswith(col, "G")]

grp = string.(df[:,1])

X = Matrix{Float64}(df[:,X_names])
G = Matrix{Float64}(df[:,G_names])
y = df[:,end]
control = Control()
control.trace = 3
#control.tol = 1e-4
#control.cov_int = (-50, 50)
#control.var_int = (0, 100000)

λs = Float64.(20:1:21)
res_matrix = Vector{Any}(undef, length(λs))
for (i, λ) in enumerate(λs)

    println("λ is $λ")
    try 
        Z = X[:,1:(end-1)]
        est = lmmlasso(X, G, y, grp, Z;
            standardize = true, penalty = "scad", 
            λ=λ, scada = 3.7, wts = fill(1.0, size(G)[2]), 
            init_coef = nothing, ψstr="ident", control=control)
    
        println("Model converged! Hurrah!")
        println("Initial number of non-zeros is $(est.init_nz)")
        println("Final number of non-zeros is $(est.nz)")
        println("Log likelihood is $(est.log_like)")
        println("BIC is $(est.bic)")
        println("Estimated L is $(est.L)")
        println("Estimated σ² is $(est.σ²)")

        #Insert the resulting fit to the results matrix but remove the
        #field that has all the data in it for memory efficiency 
        res_matrix[i] = Base.structdiff(est, (data = 1,))
    
    catch e
        println("An error occurred for λ = $λ: $e")
        res_matrix[i] = e
    end


end
