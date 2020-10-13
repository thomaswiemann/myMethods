module MyMethods

# Load dependencies
using DataFrames # for neat output 
using LinearAlgebra # for matrix operations
using Distributions # for rnd samplers
using Distributed # for parallelization

# May use export to avoid conflicts
export myLS, predict, inference, coef,
	mybootstrap, mybootstrapPAR

# Module content
## myLS
include("FUN/myLS.jl")
## mybootstrap
include("FUN/mybootstrap.jl")
include("FUN/mybootstrapPAR.jl")
## miscellaneous help functions
include("FUN/help_functions.jl")

end #MyMethods