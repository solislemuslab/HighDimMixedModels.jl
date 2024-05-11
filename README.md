# Regularized mixed-effects models in Julia
[![Documenter](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Documenter.yml) [![Runtests](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Runtests.yml/badge.svg)](https://github.com/solislemuslab/HighDimMixedModels.jl/actions/workflows/Runtests.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/)
[![codecov](https://codecov.io/github/solislemuslab/HighDimMixedModels.jl/graph/badge.svg?token=BAF8P78SUS)](https://codecov.io/github/solislemuslab/HighDimMixedModels.jl)

## Overview 
HighDimMixedModels.jl is a package for fitting regularized linear mixed-effect models on high-dimensional omics data. These models can be used to analyze hierarchical, high dimensional data, especially useful for situations in which the number of predictors exceeds the number of samples ($p > n$). 

For fitting the model, the package implements the coordinate gradient descent algorithm found in

>Schelldorfer, Jürg, Peter Bühlmann, and SARA VAN DE GEER. "Estimation for high‐dimensional linear mixed‐effects models using ℓ1‐penalization." Scandinavian Journal of Statistics 38.2 (2011): 197-214.

and 

>Ghosh, A., & Thoresen, M. (2018). Non-concave penalization in linear mixed-effect models and regularized selection of fixed effects. AStA Advances in Statistical Analysis, 102, 179-210. 

Because of its superior estimation performance, the *smoothly clipped absolute deviation* (SCAD) is the default penalty, but the $\ell_1$ penalty (LASSO) proposed in the original source is also available.


## Installation and usage 

See the [package documentation](https://solislemuslab.github.io/HighDimMixedModels.jl/dev/) for details on how to install and use the package.

## Errors and contributions

To report a bug or propose a new feature, please open an issue in the [issue tracker](https://github.com/solislemuslab/HighDimMixedModels.jl/issues). 

We welcome contributions. User who are interested in contributing code are asked to follow the instructions found in [CONTRIBUTING.md](https://github.com/solislemuslab/HighDimMixedModels.jl/blob/main/CONTRIBUTING.md)


## Citation

If you use `HighDimMixedModels.jl` in your work, we kindly ask that you cite 
```
@article{Gorstein2024,
author = {Gorstein, E. and Aghdam, R. and Sol'{i}s-Lemus, C.},
year = {2024},
title = {{HighDimMixedModels.jl: Robust High Dimensional Mixed Models across Omics Data}},
journal = {In preparation}
}
```
