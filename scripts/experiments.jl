# ]activate test
# ]dev HighDimMixedModels

#using HighDimMixedModels
using ZipFile
using CSV
using DataFrames
using DelimitedFiles
using Serialization

r = ZipFile.Reader("data/dim1000_random5_rho0.6_nz10_covid.zip")
λs = Float64.(30:20:70)
res_matrix = Array{Any}(undef, 2, length(λs))

for (i, f) in enumerate(r.files[70:71]) #First file is just the folder
    
    println("Filename: $(f.name)")
    println("File number $(i)")
    
    df = CSV.read(f, DataFrame)
    X_names = [col for col in names(df) if startswith(col, "X")]
    G_names = [col for col in names(df) if startswith(col, "G")]
    
    grp = string.(df[:,1])
    
    X = Matrix(df[:,X_names])
    G = Matrix(df[:,G_names])
    y = df[:,end]
    control = Control()
    control.trace = 3
    control.tol = 1e-2

    for (j, λ) in enumerate(λs)

        println("λ is $λ")
        try 
            est = lmmlasso(X, G, y, grp; 
                standardize = true, penalty = "scad", 
                λ=λ, scada = 3.7, wts = fill(1.0, size(G)[2]), 
                init_coef = nothing, ψstr="ident", control=control)
        
            println("Model converged! Hurrah!")
            println("Initial number of non-zeros is $(est.init_nz)")
            println("Final number of non-zeros is $(est.nz)")
            println("Log likelihood is $(est.log_like)")
            println("BIC is $(est.bic)")
            println("Estimated L is $(est.L)")

            #Insert the resulting fit to the results matrix but remove the
            #field that has all the data in it for memory efficiency 
            res_matrix[i,j] = Base.structdiff(est, (data = 1,))
        
        catch e
            println("An error occurred in file $(f.name), λ = $λ: $e")
            res_matrix[i,j] = "error"
        end


    end

end 

serialize("serial.txt", res_matrix)

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


