# Inside make.jl
push!(LOAD_PATH,"../src/")
using HighDimMixedModels
using Documenter
ENV["GKSwstype"] = "100"
makedocs(
         sitename = "HighDimMixedModels.jl",
         modules  = [HighDimMixedModels],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/solislemuslab/HighDimMixedModels.jl",
)