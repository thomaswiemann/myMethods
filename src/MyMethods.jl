module MyMethods

# Load dependencies
using DataFrames # for neat output 
using LinearAlgebra # for matrix operations
using Distributions # for rnd samplers
using Distributed # for parallelization

# May use export to avoid conflicts
export myLS, myLLR # my objects
export predict, inference, coef, coefPAR # my methods
export mybootstrap, mybootstrapPAR # my functions

# Module content
## myLS
include("FUN/myLS.jl")
## myLLR
include("FUN/myLLR.jl")
## mybootstrap
include("FUN/mybootstrap.jl")

end #MyMethods