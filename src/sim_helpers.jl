module simulations

export simulate_design
export simulate_y

using StatsBase
using Random
using Distributions


"""
Simulates fixed and random effect design matrices 

ARGUMENTS
- n :: Vector of group sizes
- q :: Number of unpenalized fixed effect parameters (includes one for intercept)
- p :: Number of penalized fixed effect parameters
- m :: Of the unpenalized predictors, number which have associated random effects
- rho :: Correlation between fixed effect predictors 

OUTPUT:
- X :: Matrix of dimension Ntot by p
- G :: Matrix of dimension Ntot by q
- Z :: Matrix of dimension Ntot by m
- grp :: grouping variable, vector of length Ntot with N distinct string values
"""
function simulate_design(;n=fill(6, 20), q=2, p=5, m=q, rho=0.2)
   
    N = sum(n) # Total number of observations
    n_pred = q + p - 1 # Total number of predictors (doesn't include intercept)

    # Create covariance matrix for design
    xcov = Matrix{Float64}(undef, n_pred, n_pred)

    for i in 1:n_pred, j in 1:n_pred
        xcov[i, j] = rho^(abs(i-j))
    end

    # Generate design matrices X, G, and Z
    dist_x = MvNormal(zeros(n_pred), xcov) 
    full_pred = rand(dist_x, N)' # Matrix of dimensions Ntot by n_pred
    
    X = hcat(fill(1, N), full_pred[:,1:(q-1)]) # Unpenalized design matrix with column of ones prepended for intercept
    Z = X[:, 1:m] # Random effect design matrix
    G = full_pred[:,q:end] # Penalized design matrix

    # Grouping variable 
    grp = string.(inverse_rle(1:length(n), n))

    return X, G, Z, grp
end


"""
Simulates response

ARGUMENTS
- X :: Unpenalized fixed effect design matrix with column for intercept
- G :: Penalized fixed effect design matrix 
- Z :: Random effect design matrix with column for intercept
- grp :: Vector of strings of same length as number of rows of X, assigning each observation to a particular group 
- βun :: Vector of length q with unpenalized fixed effect parameters
- βpen :: Vector of length p with penalized fixed effect parameters
- L :: Scalar, Vector, or Lower Triangular matrix depending on how random effect covariance structure is parameterized
- σ² :: Error variance

OUTPUT
- y :: Vector of responses
"""
function simulate_y(X, G, Z, grp, βun, βpen, L, σ²)
    
    
    #Fixed component of response
    y = X*βun + G*βpen

    groups = unique(grp) 
    g = length(groups) # Total number of groups
    m = size(Z)[2] # Number of predictors with associated random effects
    ndims(L) < 2 || (L = Matrix(L)*Matrix(L')) # If L is a matrix, it will be Cholesky factor of random effect covariance matrix
    dist_b = MvNormal(zeros(m), L) 
    b = rand(dist_b, g) # Matrix of dimensions m by N, each of whose columns is draw from dist_b
    #Add random component of response
    for (i, group) in enumerate(groups)
        group_ind = (grp .== group)
        nᵢ = sum(group_ind)
        Zᵢ = Z[group_ind,:]
        bᵢ = b[:,i]
        yᵢ = Zᵢ*bᵢ + sqrt(σ²)*randn(nᵢ)
        y[group_ind] = y[group_ind] + yᵢ
    end

    return y

end


end


