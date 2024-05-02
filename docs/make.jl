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
                "Manual" =>  ["Installation" => "man/installation.md",
                                "Example" => "man/example.md"
                ],
                "Library" => [
                   "Public Methods and Types" => "lib/public_methods.md",
                     "Internal Methods and Types" => "lib/internal_methods.md",
                ]
               ])
deploydocs(;
    repo="github.com/solislemuslab/HighDimMixedModels.jl.git"
)