## This script generates the data for the GWAS simulation study.
## The data is stored in the data/GWAS directory.
## Note that the number of features, aka the number of SNPs, is 1000.
## This means that the matrix [X G] will have 1001 columns because the first column is for intercept. 

using CSV
using DataFrames
using Random
using StatsBase
include("sim_helpers.jl") 
using Main.simulations #Exports simulate_design() and simulate_y()


# Global simulation variables
n=fill(25, 10)
grp = string.(inverse_rle(1:length(n), n))
βnz = [1,2,4,3,3,-1,5,-3,2,2]
Lid = sqrt(0.56)
Ldiag = sqrt.([3,3,2])
Lsym = [sqrt(3) 0 0; 1 sqrt(3) 0; -sqrt(2) 1 sqrt(2)]
σ² = 0.5

# Individual simulation settings
pars1 = (q=1, cov = "id")
pars2 = (q=3, cov = "id")
pars3 = (q=3, cov = "diag")
pars4 = (q=3, cov = "sym")
pars5 = (q=5, cov = "id")

settings = [pars1, pars2, pars3, pars4, pars5]

#Create setting directories for storing data files
for set in settings
    set_name = "random$(set.q)_cov$(set.cov)"
    mkdir("data/GWAS/$(set_name)")
end


# Loop through 100 data-sets
for j in 1:100 
 
    # Read in design matrix, which was created with R script using ggmix
    file_path = "data/GWAS/data$(j).csv"
    df = CSV.read(file_path, DataFrame)
    # Set seed for this data-set
    Random.seed!(j)

    # Loop through all simulation settings
    for set in settings
        
        q = set.q
        cov = set.cov
        # Make directory to store data files for this simulation setting
        set_name = set_name = "random$(q)_cov$(cov)"
        
        # Get design matrices
        X = hcat(fill(1, sum(n)), Matrix(df)[:,1:(q-1)])
        Z = X
        G = Matrix(df)[:,q:end]

        # Get fixed effect parameters
        βun = βnz[1:q]
        βpen = vcat(βnz[(q+1):end], zeros(991))

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
        colnames = ["group" ; ["X$i" for i in 1:set.q] ; ["G$i" for i in 1:(1001-q)]; "y"]
        df_to_write = DataFrame([grp X G y], colnames)

        # Define the output file name 
        file_path = "data/GWAS/$(set_name)/data$(j).csv"

        # Write the data to file
        CSV.write(file_path, df_to_write)

    end

end
