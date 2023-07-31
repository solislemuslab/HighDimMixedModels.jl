using CSV
using DataFrames
using Random
include("sim_helpers.jl") 
using Main.simulations #Exports simulate_design() and simulate_y()

# Define setting parameters
pars1 = (n=fill(6, 30), dim=500, q=1, rho = 0., nz = 5, cov = "id")
pars2 = (n=fill(6, 30), dim=500, q=1, rho = 0.6, nz = 5, cov = "id")
pars3 = (n = fill(5, 50), dim=1000, q=1, rho = 0., nz = 5, cov = "id")
pars4 = (n = fill(5, 50), dim=1000, q=1, rho = 0., nz = 10, cov = "id")
pars5 = (n = fill(5, 50), dim=1000, q=1, rho = 0.6, nz = 5, cov = "id")
pars6 = (n = fill(5, 50), dim=1000, q=1, rho = 0.6, nz = 10, cov = "id")
pars7 = (n = fill(5, 50), dim=1000, q=3, rho = 0., nz = 10, cov = "id")
pars8 = (n = fill(5, 50), dim=1000, q=3, rho = 0., nz = 10, cov = "diag")
pars9 = (n = fill(5, 50), dim=1000, q=3, rho = 0., nz = 10, cov = "sym")
pars10 = (n = fill(5, 50), dim=1000, q=3, rho = 0.6, nz = 10, cov = "id")
pars11 = (n = fill(5, 50), dim=1000, q=3, rho = 0.6, nz = 10, cov = "diag")
pars12 = (n = fill(5, 50), dim=1000, q=3, rho = 0.6, nz = 10, cov = "sym")
pars13 = (n = fill(5, 50), dim=1000, q=5, rho = 0., nz = 10, cov = "id")
pars14 = (n = fill(5, 50), dim=1000, q=5, rho = 0.6, nz = 10, cov = "id")

settings = [pars1, pars2, pars3, pars4, pars5, pars6, pars7, pars8, pars9, pars10, pars11, pars12, pars13, pars14]

# Global parameter variables
βnz1 = [1,2,4,3,3]
βnz2 = [1,2,4,3,3,-1,5,-3,2,2]
Lid = sqrt(0.56)
Ldiag = sqrt.([3,3,2])
Lsym = [sqrt(3) 0 0; 1 sqrt(3) 0; -sqrt(2) 1 sqrt(2)]
σ² = 0.5

# Loop through all simulation settings
for (i, set) in enumerate(settings)
    
    # Make directory to store data files for this simulation setting
    set_name = "dim$(set.dim)_random$(set.q)_rho$(set.rho)_nz$(set.nz)_cov$(set.cov)"
    mkdir("data/$(set_name)")

    # Set seed for this simulation setting
    Random.seed!(i)

    # Produce 100 data-sets for this simulation setting
    for j in 1:100    

        
        # Get fixed effect parameters
        βnz = (set.nz == 5) ? βnz1 : βnz2
        βun = βnz[1:set.q]
        βpen = vcat(βnz[(set.q+1):end], zeros(set.dim - set.nz))

        # Get random effect parameters
        if set.cov == "id"
            L = Lid
        elseif set.cov == "diag"
            L = Ldiag
        elseif set.cov == "sym"
            L = Lsym 
        end

        # Get design matrices and grouping structures
        X, G, Z, grp = simulate_design(; n=set.n, q = set.q, p = set.dim-set.q, m = set.q, rho = set.rho)

        # Simulate the response
        y = simulations.simulate_y(X, G, Z, grp, βun, βpen, L, σ²)

        #Convert to dataframe, giving names to columns
        colnames = ["group" ; ["X$i" for i in 1:set.q] ; ["G$i" for i in 1:(set.dim-set.q)]; "y"]
        df = DataFrame([grp X G y], colnames)

        # Define the output file name 
        file_path = "data/$(set_name)/data$(j).csv"

        # Write the data to file
        CSV.write(file_path, df)

    end

end
