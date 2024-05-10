# Installation

## Installation of Julia

Julia is a high-level and interactive programming language (like R or Matlab), but it is also high-performance (like C). To install Julia, follow instructions [here](http://julialang.org/downloads/). For a quick & basic tutorial on Julia, see [learn x in y minutes](http://learnxinyminutes.com/docs/julia/).

Editors:

- [Visual Studio Code](https://code.visualstudio.com) provides an editor
  and an integrated development environment (IDE) for Julia: highly recommended!
- You can also run Julia within a [Jupyter](http://jupyter.org) notebook
  (formerly IPython notebook).

IMPORTANT: Julia code is just-in-time compiled. This means that the
first time you run a function, it will be compiled at that moment. So,
please be patient! Future calls to the function will be much much
faster. Trying out toy examples for the first calls is a good idea.


## Installation of the package 
To install `HighDimMixedModels`, type in the Julia REPL
```julia
]
add HighDimMixedModels
```

The installation can take a few minutes, be patient. The package has dependencies such as [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) and [Lasso.jl](https://juliastats.org/Lasso.jl/stable/) (see the `Project.toml` file for the full list), but everything is installed automatically. 



