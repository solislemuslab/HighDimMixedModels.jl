using HighDimMixedModels
using Serialization
using CSV
using DataFrames

# matrix to store result for best lambda
coef_cols = 128

best_results = Matrix{Any}(undef, 1200*5, 9+25+coef_cols)

#Get names of serialized files
dir_path =  "sim_results/OTU/spring/" 
setting_names = readdir(dir_path)
setting_names = setting_names[endswith.(setting_names, "results.txt")]

#Iterate through each setting
global i = 1 #Index of next row to fill in in best_results
for setting_name in setting_names
    setting = open(dir_path*setting_name, "r")
    all_setting_results = deserialize(setting)

    true_nz = 11
    #Iterate through each data file in the setting
    for row in eachrow(all_setting_results)

        data_id = replace(row[1], ".csv" => "")
        data_results = row[2:end]
        # Filter out fits that resulted in errors
        data_good_results = filter(x -> !isa(x, String), data_results)
        # Sort rows and extract best results
        data_sorted_results = sort(data_good_results, by = x -> x.bic)
        k = min(length(data_sorted_results), 2)
        data_best_results = data_sorted_results[1:k]

        #Add a single row to the best_results matrix for each row in data_best_results
        for result in data_best_results

            # Get number of true positives
            nz_ind = findall(result.fixef .!= 0)
            tp = length(intersect(nz_ind, 1:true_nz))

            # Remove suffix from setting_name
            setting_name = replace(setting_name, ".txt" => "")

            # Create row to get added to matrix of best results
            added_row = [setting_name, data_id, result.λ, result.log_like, result.aic, result.bic, result.nz, tp, sqrt(result.σ²)]
            
            # Add psi
            vec_ψ = vec(result.ψ)
            added_row = [added_row; vec_ψ; fill(missing, 25-length(vec_ψ))]

            # Add beta estimates
            added_row = [added_row; result.fixef]

            # Add row to matrix of best results
            best_results[i,:] = added_row
            global i = i+1
        end
    end
    close(setting)
end

#Create dataframe
colnames = ["setting" ; "data_id"; "lambda"; "loglike"; "aic"; "bic"; "n_nz"; "tp"; "sigma"; 
            ["psi_$i" for i in 1:25]; ["beta_$i" for i in 1:coef_cols]]
best_results = DataFrame(best_results[1:i-1,:], colnames)

# Define the output file name 
file_path = "sim_results/OTU/spring/best_results.csv" 

# Write the data to file
CSV.write(file_path, best_results)