##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# M
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
mutable struct highDimMat{T, S<:AbstractVecOrMat}
    M::S
end 


# constructor for highDimMat M,
function highDimMat(M::AbstractVecOrMat{T}) where {T}
    if size(M,1) >= size(M,2)
        @warn "n >= p in high dimensional matrix"
    end
    highDimMat{T, typeof(M)}(M)
end


"""
# constructor for highDimMat M,
function highDimMat(M::AbstractMatrix)
    if size(M,1) >= size(M,2)
        @warn "n >= p in high dimensional matrix"
    end
    return highDimMat{eltype(M), typeof(M)}(M)
end
"""

Base.copyto!(A::highDimMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.M, src)

Base.eltype(::highDimMat{T}) where {T} = T

Base.getindex(A::highDimMat{T}, i::Int, j::Int) where {T} = getindex(A.M, i, j)

Base.size(A::highDimMat{T}) where {T} = size(A.M)

Base.size(A::highDimMat{T}, i::Integer) where {T} = size(A.M, i)
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# X
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
mutable struct XMat{T, S<:AbstractVecOrMat}
    X::S
end 

# constructor for XMat X,
function XMat(X::AbstractVecOrMat{T}) where {T}
    if rank(X) < size(X,2)
        @warn "fixed effect matrix is not of full rank"
    end

    if size(X,1) < size(X,2)
        @warn "n < p in covariate matrix X"
    end
    XMat{T, typeof(X)}(X)
end

Base.copyto!(A::XMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.X, src)

Base.eltype(::XMat{T}) where {T} = T

LinearAlgebra.rank(A::XMat{T}) where {T} = rank(A.X)

isfullrank(A::XMat{T}) where {T} = rank(A) == size(A.X,2)

Base.getindex(A::XMat{T}, i::Int, j::Int) where {T} = getindex(A.X, i, j)

#Base.adjoint(A::XMat{T}) = Adjoint(A)

Base.size(A::XMat{T}) where {T} = size(A.X)

Base.size(A::XMat{T}, i::Integer) where {T} = size(A.X, i)

## define new infix binary operator?

function Base.:*(A::highDimMat{T}, B::XMat{T}) where{T}
    A.M*B.X
end


#*(A::highDimMat{T, AbstractMatrix{T}}, B::XMat{T, AbstractMatrix{T}}) where{T} = A.M*B.X

*(A::highDimMat{Int64, Matrix{Int64}}, B::XMat{Int64, Matrix{Int64}}) where{T} = A.M*B.X

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# Z
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
abstract type AbstractReMat{T} <: AbstractMatrix{T} end

mutable struct ReMat{T,S} <: AbstractMatrix{T}
    #trm # the grouping factor as a `StatsModels.CategoricalTerm`   ##????
    Z::Matrix{T}
end

# constructor for XMat X,
function ReMat(Z::AbstractMatrix{T}) where {T}
    ReMat{T, typeof(Z)}(Z)
end



LinearAlgebra.rank(A::ReMat) = rank(A.Z)

isfullrank(A::ReMat) = rank(A) == size(A.Z,2)

Base.getindex(A::ReMat, i::Int, j::Int) = getindex(A.Z, i, j)

Base.size(A::ReMat) = size(A.Z)


##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# highDimMixedModel
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
"""
    highDimMixedModel
High dim mixed-effects model representation
## Fields
* `formula`: the formula for the model
* `optsum`: an [`OptSummary`](@ref) object
## Properties
* `??` or `theta`: the covariance parameter vector used to form ??
* `??` or `beta`: the fixed-effects coefficient vector
* `??` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `??`
* `??` or `sigma`: current value of the standard deviation of the per-observation noise
* `b`: random effects on the original scale, as a vector of matrices
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of ??
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
mutable struct highDimMixedModel{T<:AbstractFloat}  <: MixedModel{T}
    formula::FormulaTerm
    M::highDimMat{T}
    X::XMat{T}
    Z::ReMat{T}
    y::Vector{T}
    optsum::Union{OptSummary{T}, Nothing}
end


""" (1)
    highDimMixedModel
dealing with the number of columns in M and X scenario, assume intercept alwalys exists (@formula: y ~ 1 + ... )
# @ numOfHDM: number of HDM variables
# @ numOfHDM: number of X variables

This function assumes the variables in dataframe is arranged in order HDM, X , Z(ReMat).
 The issue here is matrix of random effect is without intercept (set to zero constrain)

"""
function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    contrasts::Dict{Symbol, UnionAll},
    numOfHDM::Int64,
    numOfXMat::Int64
) where{T}
    for i in 1:size(df,2)
        if(isa(df[1,i], Number))
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    sch = schema(df,contrasts)
    form = apply_schema(f, sch)
    y, pred = modelcols(form, df);
    MXZ = pred[:,2:size(pred,2)]  ## get rid of intercept
    M = highDimMat(MXZ[:,1:numOfHDM])
    intercept = pred[:,1]
    X = XMat(hcat(reshape(intercept, size(intercept,1),1), MXZ[:, (numOfHDM + 1):(numOfHDM + numOfXMat)]))  ## concatenate intercept with X
    Z = ReMat(MXZ[:, (numOfHDM + numOfXMat + 1):size(MXZ,2)])
    
    #return highDimMixedModel{T<:AbstractFloat}(form, M, X, Z, y)
    return highDimMixedModel{Float64}(form, M, X, Z, y, nothing)
end


""" (2)
    highDimMixedModel(...)
   Private constructor for a highDimMixedModel.
To construct a model, you only need the formula (`f`), data(`df`), contrasts and indices of M,X,Z

# the ID given here must be coincide with the ID in formula
# The issue here is matrix of random effect is without intercept (set to zero constrain)

# @ idOfHDM: id of variables corresponding to matrix M, NOTE: it's the ID in hte formula! # [1,2]
# @ idOfXMat: same as before
# @ idOfReMat: same as before
"""



""" not useful right now
private help function for select indices for remaining random effect matrix
    preTerms is Vector(1:length(form.rhs.terms))

_filt(x, preTerms) = !(x in preTerms)

preTerms = Vector(1:length(form.rhs.terms))

"""

function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    contrasts::Dict{Symbol, UnionAll},
    idOfHDM::Union{Int,AbstractArray{Int64,1}}, # [1,2]
    idOfXMat::Union{Int,AbstractArray{Int64,1}},
    idOfReMat::Union{Int,AbstractArray{Int64,1}}
) where {T} 
    for i in 1:size(df,2)
        if(isa(df[1,i], Number))
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    sch = schema(df, contrasts)
    form = apply_schema(f, sch)
    y, pred = modelcols(form, df);
    terms = form.rhs.terms
    M = highDimMat(modelmatrix(terms[idOfHDM],df))
    X = XMat(modelmatrix(terms[idOfXMat],df))

    #preTerms = Vector(1:length(form.rhs.terms))
    #idOfReMat = preTerms[preTerms .??? Ref(vcat(idOfHDM, idOfXmat))]
    Z = ReMat(modelmatrix(terms[idOfReMat],df))

    return highDimMixedModel{Float64}(form, M, X, Z, y, nothing)

end

""" (3)
    highDimMixedModel(...)
   Private constructor for a highDimMixedModel.
To construct a model, you only need the formula (`f`), data(`df`), contrasts and names of columns of M,X,Z

!!! note
    This method requires the order of terms in formula corresponds to the order of names in df
"""
function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    contrasts::Dict{Symbol, UnionAll},
    nameOfHDM::Union{AbstractString,AbstractArray{<:AbstractString,1}}, # ["a","b"]
    nameOfXMat::Union{AbstractString,AbstractArray{<:AbstractString,1}},
    nameOfReMat::Union{AbstractString,AbstractArray{<:AbstractString,1}}
) where {T} 
    """
    for i in 1:size(df,2)
        if(eltype(df[:,i]) == Int64)
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    """
    for i in 1:size(df,2)
        if(isa(df[1,i], Number))
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    sch = schema(df, contrasts)
    form = apply_schema(f, sch)
    y, pred = modelcols(form, df);
    terms = form.rhs.terms

    if(typeof(form.rhs.terms[1]) <: InterceptTerm)
        ### This line is due to 0 intercept term in formula
        terms = terms[2:length(terms)]
    end

    ## get id by names
    namesOfVar = names(df)[2:length(names(df))] ## extract variable names
    if typeof(nameOfHDM) == String idOfHDM = findall(x -> x == nameOfHDM, namesOfVar)
    else idOfHDM = findall(x -> x in nameOfHDM, namesOfVar) ; end

    if typeof(nameOfXMat) == String idOfXMat = findall(x -> x == nameOfXMat, namesOfVar)
    else idOfXMat = findall(x -> x in nameOfXMat, namesOfVar) ; end
    
    if typeof(nameOfReMat) == String idOfReMat = findall(x -> x == nameOfReMat, namesOfVar)
    else idOfReMat = findall(x -> x in nameOfReMat, namesOfVar) ; end
    
    M = highDimMat(modelmatrix(terms[idOfHDM],df))
    X = XMat(modelmatrix(terms[idOfXMat],df))

    #preTerms = Vector(1:length(form.rhs.terms))
    #idOfReMat = preTerms[preTerms .??? Ref(vcat(idOfHDM, idOfXmat))]
    Z = ReMat(modelmatrix(terms[idOfReMat],df))

    return highDimMixedModel{Float64}(form, M, X, Z, y, nothing)

end

""" (4)
    highDimMixedModel(...)
   Private constructor for a highDimMixedModel.
To construct a model, you only need the formula (`f`), data(`df`), names of columns of M,X,
this function utilize the apply_schema() from MixedModel, so it can parse the y ~ 0 + a + b + (1|c) as mixed model
The modelcols/modelmatrix in StatsModels.jl cannot parse (1|c), need more info to distinguish between matrix M and X
    
!!! note
    This method requires the order of terms in formula corresponds to the order of names in df
"""
function highDimMixedModel(
    f::FormulaTerm,
    df::DataFrame,
    numOfHDM::Int64,
    #numOfXMat::Int64
) where {T} 
    """
    for i in 1:size(df,2)
        if(eltype(df[:,i]) == Int64)
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    """
    for i in 1:size(df,2)
        if(isa(df[1,i], Number))
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end

    ## below utilize apply_schema() from MixedModel, the reason to keep above is we still need variable names 
    ## to identify M and X
    ## use apply_schema to get random effect matrix
    sch = schema(df)
    form = apply_schema(f, sch, highDimMixedModel)
    #preTerms = Vector(1:length(form.rhs.terms))
    #idOfReMat = preTerms[preTerms .??? Ref(vcat(idOfHDM, idOfXmat))]
    y, pre = modelcols(form,df)
    FixedEffectMatrix,ZMatrix = pre  #modelmatrix(form,df)
    M = highDimMat(FixedEffectMatrix[:,1:numOfHDM])
    X = XMat(FixedEffectMatrix[:, (numOfHDM + 1):size(FixedEffectMatrix,2)])
    Z = ReMat(ZMatrix)

    return highDimMixedModel{Float64}(form, M, X, Z, y, nothing)

end



""" (5)
    highDimMixedModel(...)
   Private constructor for a highDimMixedModel.
   This function is a generalization of constructor when we have high dimensional data, in this setting, we won't need formula(since it's too long),
        we will need id of each random variable, then we will directly use id of variable of M and X make the matrix M and X, 
        use tricks in (4) to make Z
    

# the ID given here must be coincide with the ID in formula
# The issue in (1) is matrix of random effect is without intercept (set to zero constrain), we try to tackle it here using tricks in (4)

# @ nameOfy: the variable name of response for constructing formula # ["a", "b"]
# @ idOfReMat: same as before

# @ idOfHDM: id of variables corresponding to matrix M, NOTE: it's the ID in the df! # [1,2]
# @ idOfXMat: same as before

"""

function highDimMixedModel(
    df::DataFrame,
    nameOfy::Union{AbstractString,AbstractArray{<:AbstractString,1}}, # ["a", "b"]
    nameOfReMat::Union{AbstractString,AbstractArray{<:AbstractString,1}},
    idOfHDM::Union{Int,AbstractArray{Int64,1}}, # [1,2]
    idOfXMat::Union{Int,AbstractArray{Int64,1}},
    #contrasts = Dict(term(nameOfReMat) => CategoricalTerm)
) where {T} 
    for i in 1:size(df,2)
        if(isa(df[1,i], Number))
            df[!,i] = convert(Vector{Float64},df[:,i])
        end
    end
    
    M = highDimMat(Matrix(df[:,idOfHDM]))
    X = XMat(Matrix(df[:,idOfXMat]))

    # The function doesn't need formula to construct M and X, so the formula constucted here is only for extracting Z, 
    # doesn't matter we have intercept or not
    # need to transfer nameOfy and nameOfReMat to term due to error: KeyError: key nameOfReMat not found
    # https://juliastats.org/StatsModels.jl/stable/api/#StatsModels.term
    # need to transform nameOfReMat from string to categorical ? 

    #transform!(df, :trt => categorical, renamecols=false) # this is need to avoid error in term(nameOfReMat)
    f = @formula(term(nameOfy) ~ (1|term(nameOfReMat)))
    print(nameOfy)
    #sch = schema(df, contrasts)
    sch = schema(df)
    form = apply_schema(f, sch, highDimMixedModel)
    
    y, pre = modelcols(form,df)
    FixedEffectMatrix,ZMatrix = pre 
    Z = ReMat(ZMatrix)

    return highDimMixedModel{Float64}(form, M, X, Z, y, nothing)

end


##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# fit
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
"""
function fit(
    ::Type{highDimMixedModel},
    f::FormulaTerm,
    df::DataFrame;
    contrasts,
    progress,
    REML,
    idOfHDM::Union{Int,AbstractArray{Int64,1}}, # [1,2]
    idOfXMat::Union{Int,AbstractArray{Int64,1}},
    idOfReMat::Union{Int,AbstractArray{Int64,1}}
) return fit!(highDimMixedModel(f, df, contrasts, idOfHDM, idOfXMat, idOfReMat), progress, REML)
end
"""
function fit(
    HMM::highDimMixedModel{T};
    verbose::Bool=true,
    REML::Bool=true,
    alg = :LN_COBYLA,
) where{T}
    return fit!(HMM, verbose = verbose, REML = REML, alg = alg)
end


function fit!(HMM::highDimMixedModel{T}; verbose::Bool=true, REML::Bool=true, alg = :LN_COBYLA) where {T}
    n = size(HMM.M, 1)
    A = hcat(HMM.M.M, HMM.X.X)
    P = I - A*inv(transpose(A)*A)*transpose(A)
    u,s,v = svd(P)
    r = size(HMM.M,2) + size(HMM.X,2)  # simplify: assume fixed effect full rank
    #C = randn((n-r),n)
    C = transpose(u[:,1:(n-r)]) 
    K = C*P
    Z = HMM.Z.Z
    y = HMM.y
    # C can be any full rank matrix with size n,r, e.g. randn(n,r)

    ## add optsum
    # init para
    sigma = [2200.0,14.0]
    lbd = [0.0; 0.0]
    optsum = OptSummary(Float64.(sigma), lbd, alg; ftol_rel=T(1.0e-15), ftol_abs=T(1.0e-8), xtol_rel = 1e-10)
    optsum.REML = REML
    
    ## init opt based on optsum
    opt = Opt(optsum)

    function negLogLik(sigma::Vector{Float64}, g::Vector{Float64})
        n = length(y)
        Sigma = sigma[1]*Z*transpose(Z) + sigma[2]*diagm(ones(n))
        negLog = 1/2*log(det(K*Sigma*transpose(K))) + 1/2*transpose(y)*transpose(K)*inv(K*Sigma*transpose(K))*K*y
        #println("OPT: parameter $(sigma) || objective eval $(negLog)")
        @show sigma
        @show negLog

        return negLog
    end

    println("The initial object value is $(negLogLik(Float64.(sigma), [1.0,1.0]))")

    opt.min_objective = negLogLik
    optsum.finitial = negLogLik(optsum.initial, [1.0,1.0])  # the second field not useful right now

    

    if verbose println("OPTBL: starting point $(sigma)") ; end    # to stdout

    (optf,optx,ret) = optimize(opt, sigma)
    

    Sigma = optx[1]*Z*transpose(Z) + optx[2]*diagm(ones(n))
    beta = inv(transpose(A)*inv(Sigma)*A)*transpose(A)*inv(Sigma)*y
    betaM = beta[1:size(HMM.M,2)]
    betaX = beta[(size(HMM.M,2) + 1): size(beta)[1]]

    optsum.feval = opt.numevals
    optsum.final = optx
    optsum.fmin = optf
    optsum.returnvalue = ret

    if verbose println("got $(round(optf, digits=5)) at $(round.(optx, digits=5)) after $(opt.numevals) iterations (returned $(ret))") ; end

    HMM.optsum = optsum

    return optx, betaM, betaX, optsum

end


##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# Base.getproperty
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============

function Base.getproperty(HMM::highDimMixedModel{T}, s::Symbol) where{T}
    if s == :?? || s == :sigma
        get??(HMM)
    elseif s == :?? || s ==:beta
        get??(HMM)
    elseif s == :??M || s == :betaM
        get??M(HMM)
    elseif s == :??X || s == :betaX
        get??X(HMM)
    else
        getfield(HMM,s)
    end
end

function get??(HMM::highDimMixedModel{T}) where{T}
    return HMM.optsum.final
end

function get??(HMM::highDimMixedModel{T}) where{T}
    n = size(HMM.M, 1)
    A = hcat(HMM.M.M, HMM.X.X)
    Z = HMM.Z.Z
    y = HMM.y
    optx = HMM.sigma
    Sigma = optx[1]*Z*transpose(Z) + optx[2]*diagm(ones(n))
    beta = inv(transpose(A)*inv(Sigma)*A)*transpose(A)*inv(Sigma)*y
    return beta
end

function get??M(HMM::highDimMixedModel{T}) where{T}
    beta = get??(HMM)[1:size(HMM.M,2)]
    return beta[1:size(HMM.M,2)]
end

function get??X(HMM::highDimMixedModel{T}) where{T}
    beta = get??(HMM)[1:size(HMM.M,2)]
    return beta[(size(HMM.M,2) + 1): size(beta)[1]]
end

##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============
# Base.show
##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============##==============

function Base.show(io::IO, ::MIME"text/plain", HMM::highDimMixedModel)
    if isnothing(HMM.optsum)
        return
    end
    REML = HMM.optsum.REML
    println(io, "Linear mixed model fit by ", REML ? "REML" : "not implemented yet")
    println(io, " ", HMM.formula)
    obj = HMM.optsum.fmin
    if REML
        println(io, " REML criterion at convergence: ", obj)
    else
        println("Only implemented REML at this time")
    end

    println("??_z : $(HMM.sigma[1])")
    println("?? : $(HMM.sigma[2])")
    println("??_M : $(HMM.betaM)")
    println("??_X : $(HMM.betaX)")
    
end













