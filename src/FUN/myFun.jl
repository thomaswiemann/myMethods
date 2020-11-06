# Additional smaller functions

# Function for empirical covariance matrix b/w rnd variables
function myCov(X; corrected=true)
    # Data parameters
    N = size(X,1)
    # Calculate covariance matrix
    EXX = X'*X./N
    EX =  mapslices(mean, X, dims=1)
    # Dof adjustment
    if corrected
        Cov = (EXX - EX'*EX)*(N/(N-1))
    else
        Cov = EXX - EX'*EX
    end
    # Return covariance matrix
    return Cov
end #MYCOV

# Method for distance calculation b/w two points
function myDist(x::Float64, y::Float64; metric = "Manhattan")
	if metric == "Manhattan"
		return abs(x-y)
	end
end #MYDIST

# Method for distance calculation b/w two vectors
function myDist(x::Array{Float64,1},y::Array{Float64,1}; metric = "Mahalanobis",
	S = nothing, S_inv = nothing)
	# Check whether S_inv is passed
	if isnothing(S_inv)
		S_inv = inv(S)
	end
	# Calculate distance
	if metric == "Mahalanobis" || metric == "InvVariance"
		return sqrt((x-y)'*S_inv*(x-y))
	end
end #MYDIST

# Method for distance calculation b/w rows of matrix X and vector y
function myDist(X::Array{Float64,2},y::Array{Float64,1}; metric = "Mahalanobis",
	S = nothing)
	# Check whether S needs to be calculated
	if metric == "Mahalanobis" && isnothing(S)
		S = myCov(X)
	elseif metric == "InvVariance" && isnothing(S)
		S = myCov(X)
		S = diagm(S[diagind(S)])
	end
	# Calculate S_inv
	S_inv = inv(S)
	
	# Broadcast myDist to rows of X
	return mapslices(x -> myDist(x, y, metric=metric, S_inv=S_inv), X, dims=2)[:]
end #MYDIST

# Create dummy matrix from discrete variables
# to do: implement sparse matrix 
function myDummify(x, sparse=false)
    # Check number of unique values in x
    uni_x = sort(unique(x))
    nx = length(uni_x)
    # Construct matrix of indicators
    X = zeros(length(x), nx)
    for k in 1:nx
        indx = findall(x.==uni_x[k])
        X[indx, k] .= 1
    end
    # Return X
    return X
end