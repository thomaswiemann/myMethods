module MyMethods

# Load dependencies
using DataFrames # for neat output 
using LinearAlgebra # for matrix operations
using Distributions # for rnd samplers
using Distributed # for parallelization
using Statistics # for quantile function

# May use export to avoid conflicts
export myLS, mySieve, myLLR, myKNN, myMatch # my objects
export predict, inference, coef, coefPAR, R2 # my methods
export mybootstrap, mybootstrapPAR, reduce_boot # my functions
export myCov, myDist # smaller functions


# Module content
## myLS
include("FUN/myLS.jl")
## mySieve
include("FUN/mySieve.jl")
## myLLR
include("FUN/myLLR.jl")
## myKNN
include("FUN/myKNN.jl")
## myMatch
include("FUN/myMatch.jl")
## mybootstrap
include("FUN/mybootstrap.jl")
## basic statistical functions
include("FUN/myFun.jl")

end #MyMethods