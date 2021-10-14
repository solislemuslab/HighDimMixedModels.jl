module highDimMM

#using MixedModels: modelmatrix
using Base: Float64, AbstractVecOrTuple
using DataFrames: Dict
#using LinearAlgebra: AbstractMatrix, include
using StatsModels
using LinearAlgebra
using DataFrames
using MixedModels
using NLopt

import Base: *

abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

include("bricks.jl")
#include("bricks_Number.jl")


export highDimMixedModel, modelmatrix, fit


end