module HighDimMixedModels

# Write your package code here.

#using MixedModels: modelmatrix
using Base: Float64, AbstractVecOrTuple
using DataFrames: Dict
#using LinearAlgebra: AbstractMatrix, include
using StatsModels
using LinearAlgebra
using DataFrames
using MixedModels
using NLopt
using CategoricalArrays

import Base: *
import MixedModels: fit, fit!

# abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

include("bricks.jl")
#include("bricks_Number.jl")


export highDimMixedModel, 
    modelmatrix, 
    fit, 
    fit!,
    highDimMat,
    XMat,
    ReMat


end