using HighDimMixedModels
using ZipFile
using CSV
using DataFrames
using DelimitedFiles
using Serialization

# r = ZipFile.Reader("data/GWAS/random3_covsym.zip")
λs = Float64.(30:1:40)
res_matrix = Array{Any}(undef, 100, length(λs)+1)
data_dir = "data/OTU/random1_covid"
file_names = readdir(data_dir)
filter!(x -> occursin("rs_data_LinShi", x), file_names)
println(length(file_names))

# for (i, f) in enumerate(r.files[2:end]) #First file is just the folder
for (i, f) in enumerate(file_names)
    
    println("Filename: $f")
    println("File number $i")
    #First column of results matrix is the file name
    res_matrix[i, 1] = Base.basename(f)


    df = CSV.read("$data_dir/$f", DataFrame)
    X_names = [col for col in names(df) if startswith(col, "X")]
    G_names = [col for col in names(df) if startswith(col, "G")]
    
    grp = string.(df[:,1])
    
    X = Matrix{Float64}(df[:,X_names])
    G = Matrix{Float64}(df[:,G_names])
    y = df[:,end]
    control = Control()
    control.trace = 3
    #control.tol = 1e-4
    #control.cov_int = (-5, 5)
    #control.var_int = (0, 1000)

    for (j, λ) in enumerate(λs)

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
            res_matrix[i,j+1] = Base.structdiff(est, (data = 1,weights=1, fitted=1))
        
        catch e
            println("An error occurred in file $(f), λ = $λ: $e")
            res_matrix[i,j+1] = "error"
        end


    end

end 

serialize("sim_results/OTU/LinShi/yesstand_rs_random1_covid_scad-results.txt", res_matrix)

# f = r.files[2]
# println("Filename: $(f.name)")
# rx_q = r"random(\d+)"
# q = match(rx_q, f.name).captures[1]
# q = parse(Int, q)
# df = CSV.read(f, DataFrame)
# X_names = [col for col in names(df) if startswith(col, "X")]
# G_names = [col for col in names(df) if startswith(col, "G")]

# grp = string.(df[:,1])

# X = Matrix(df[:,X_names])
# G = Matrix(df[:,G_names])
# y = df[:,end]
# control = Control()
# control.trace = 3

# est = lmmlasso(X, G, y, grp; 
#     standardize = false, penalty = "scad", 
#     λ=50.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
#     init_coef = nothing, ψstr="ident", control=control)

# est2 = lmmlasso(X, G, y, grp; 
#     standardize = false, penalty = "scad", 
#     λ=55.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
#     init_coef = nothing, ψstr="ident", control=control)

# est3 = lmmlasso(X, G, y, grp; 
#     standardize = false, penalty = "scad", 
#     λ=70.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
#     init_coef = nothing, ψstr="ident", control=control)

# est4 = lmmlasso(X, G, y, grp; 
#     standardize = true, penalty = "scad", 
#     λ=55.0, scada = 3.7, wts = fill(1.0, size(G)[2]), 
#     init_coef = nothing, ψstr="ident", control=control)


