## This script generates the data for the OTU simulation study.
## The data is stored in the data/OTU directory.

using CSV
using DataFrames
using Random
using StatsBase
include("sim_helpers.jl") 
using Main.simulations #Exports simulate_design() and simulate_y()


# Global simulation variables
n=fill(12, 10)
grp = string.(inverse_rle(1:length(n), n))
βnz = [1,2,4,3,3,-1,5,-3,2,2]
β_grp = -1
Lid = sqrt(.56)
Ldiag = sqrt.([.56,1,1])
Lsym = [sqrt(.56) 0 0; sqrt(.5) sqrt(.5) 0; sqrt(.1) -sqrt(.8) sqrt(.1)]
σ² = .5

# Individual simulation settings
pars1 = (q=1, cov = "id")
pars2 = (q=3, cov = "id")
pars3 = (q=3, cov = "diag")
pars4 = (q=3, cov = "sym")
pars5 = (q=5, cov = "id")

#Can skip some settings if too hard for microbiome data
settings = [pars1, pars2, pars3, pars4, pars5]

# # Create setting directories for storing data files
# for set in settings
#     set_name = "random$(set.q)_cov$(set.cov)"
#     mkdir("data/OTU/$(set_name)")
# end

# # Type of correlation between taxa
X_cor = ["band", "cluster", "scale_free", "erdos_renyi", "hub", "block"]

Random.seed!(1)
for cor in X_cor
# Loop through 100 data-sets
    for j in 1:100 
    
        # Read in design matrix, which was created with R script 
        # Replace rs with lr for log ratio data instead
        file_path = "data/OTU/spring/data$(cor)_$(j).csv"
        df = CSV.read(file_path, DataFrame)
        df = Matrix(df)

        # Loop through all simulation settings
        for set in settings

            q = set.q
            cov = set.cov
            # Make directory to store data files for this simulation setting
            set_name = "random$(q)_cov$(cov)"

            # Get design matrices
            Z = hcat(fill(1, sum(n)), df[:,1:(q-1)])
            #Generate a variable measured a the group level and add it to X
            group_var = randn(length(n))[parse.(Int, grp)]
            X = [Z group_var]
            #Remove the last column for microbiome data because of simplex constraint
            G = df[:,q:(end-1)] 
            p = size(G, 2)
            # Get fixed effect parameters
            βun = [βnz[1:q]; β_grp] 
            βpen = vcat(βnz[(q+1):end], zeros(p+q-10))

            # Get random effect parameters
            if set.cov == "id"
                L = Lid
            elseif set.cov == "diag"
                L = Ldiag
            elseif set.cov == "sym"
                L = Lsym 
            end

            # Simulate the response
            y = simulations.simulate_y(X, G, Z, grp, βun, βpen, L, σ²)

            #Convert to dataframe, giving names to columns
            colnames = ["group" ; ["X$i" for i in 1:set.q+1] ; ["G$i" for i in 1:p]; "y"]
            df_to_write = DataFrame([grp X G y], colnames)

            # Define the output file name 
            file_path = "data/OTU/$(set_name)/data_$(cor)_$(j).csv"

            # Write the data to file
            CSV.write(file_path, df_to_write)

        end

    end
end