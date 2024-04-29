# HighDimMixedModels
[![Documenter](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Documenter.yml) [![Runtests](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Runtests.yml/badge.svg)](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Runtests.yml)
[![codecov](https://codecov.io/github/solislemuslab/HighDimMixedModels.jl/graph/badge.svg?token=BAF8P78SUS)](https://codecov.io/github/solislemuslab/HighDimMixedModels.jl)

HighDimMixedModels.jl is a package for fitting penalized linear mixed-effect models. It is a Julia implementation of the coordinate gradient descent algorithm proposed in

>Schelldorfer, Jürg, Peter Bühlmann, and SARA VAN DE GEER. "Estimation for high‐dimensional linear mixed‐effects models using ℓ1‐penalization." Scandinavian Journal of Statistics 38.2 (2011): 197-214.

The default penalty that is applied is the *smoothly clipped absolute deviation* (SCAD) penalty, as proposed in

>Ghosh, A., & Thoresen, M. (2018). Non-concave penalization in linear mixed-effect models and regularized selection of fixed effects. AStA Advances in Statistical Analysis, 102, 179-210. 

although the LASSO is also available. 

This package can be used to analyze (fit linear mixed-effect regression models to) clustered data, and the penalized approach to estimation is especially useful for situations in which the number of predictors exceeds the number of samples ($p > n$). 
