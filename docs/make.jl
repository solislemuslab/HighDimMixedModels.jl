# Inside make.jl
push!(LOAD_PATH,"../src/")
using HighDimMixedModels
using Documenter
ENV["GKSwstype"] = "100"
makedocs(
         sitename = "HighDimMixedModels.jl",
         modules  = [HighDimMixedModels],
         pages=[
                "Home" => "index.md",
                "Manual" =>  [
                    "Installation" => "man/installation.md",
                    "Input Data" => "man/inputdata.md",
                    "Model Fitting" => "man/model_fit.md",
                ],
                "Library" => [
                   "Public Methods and Types" => "lib/public_methods.md"
                ]
               ])
deploydocs(;
    repo="github.com/solislemuslab/HighDimMixedModels.jl.git"
)