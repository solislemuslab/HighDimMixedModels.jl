using Revise
using HighDimMixedModels
using ZipFile
using CSV
using DataFrames
using DelimitedFiles
using Serialization


r = ZipFile.Reader("data/GWAS/random3_covdiag.zip")


λs = Float64.(55:5:70)
res_matrix = Array{Any}(undef, 1, length(λs) + 1)
file = filter(f -> f.name == "random3_covdiag/data23.csv", r.files)  

for (i, f) in enumerate(file) 
    #for (i, f) in enumerate(file_names)
    
    println("Filename: $(f.name)")
    println("File number $i")
    #First column of results matrix is the file name
    res_matrix[i, 1] = f.name


    df = CSV.read(f, DataFrame)
    X_names = [col for col in names(df) if startswith(col, "X")]
    G_names = [col for col in names(df) if startswith(col, "G")]

    grp = string.(df[:, 1])

    X = Matrix{Float64}(df[:, X_names])
    G = Matrix{Float64}(df[:, G_names])
    y = df[:, end]
    control = Control()
    control.trace = 3
    control.tol = 1e-4
    #control.cov_int = (-5, 5)
    #control.var_int = (0, 1000)

    for (j, λ) in enumerate(λs)

        println("λ is $λ")
        try
            est = lmmlasso(X, G, y, grp;
                standardize=true, penalty="scad",
                λ=λ, scada=3.7, wts=fill(1.0, size(G)[2]),
                init_coef=nothing, ψstr="diag", control=control)

            println("Model converged! Hurrah!")
            println("Initial number of non-zeros is $(est.init_nz)")
            println("Final number of non-zeros is $(est.nz)")
            println("Log likelihood is $(est.log_like)")
            println("BIC is $(est.bic)")
            println("Estimated L is $(est.L)")
            println("Estimated σ² is $(est.σ²)")

            #Insert the resulting fit to the results matrix but remove the
            #field that has all the data in it for memory efficiency 
            res_matrix[i, j+1] = Base.structdiff(est, (data=1, weights=1, fitted=1, resid=1))

        catch e
            println("An error occurred in file $(f), λ = $λ: $e")
            res_matrix[i, j+1] = "error"
        end


    end

end