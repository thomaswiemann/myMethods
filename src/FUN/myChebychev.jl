# Define myChebychev object
struct myChebychev
    θ::Array{Float64} # Chebychev coefficients
    interval::Array{Float64} # interval of approximation
    M # number of approximation points
	n # degree of polynomial
    
    # Define constructor function
    function myChebychev(FUN,  interval; M = 7, n = 3)
    # Data parameters
    D = size(interval, 1)
    
    # Get approximation points
    ν = -cos.(((2collect(1:M) .- 1) * pi) / 2M)
    ν = collect(ν for d in 1:D)
    ν = reduce(hcat, 
        collect.(collect(zip(vec(collect(Base.product(ν...)))...))))

    # Calculate Chebychev polynomials
    T = calc_chebychevpoly(ν, n)

    # Calculate function evaluations at X
    Z = translate_from_chebychev(ν, interval)
    y = mapslices(x -> FUN(x), Z, dims = 2)

    # Calculate Chebychev coefficients
	K = (n+1)^D
    diag_TT = fill(0.0, K)
    for j in 1:K
        diag_TT[j] = T[:, j]' * T[:, j]
    end
    θ = (T' * y) ./ diag_TT
    
    # Return output
    new(θ, interval, M, n)
    end #MYCHEBYCHEV
    
end #MYCHEBYCHEV

# Methods for objects of type myChebychev

## Prediction function for myChebychev object
function predict(obj::myChebychev; data)
    # Translate input data
	ν = translate_to_chebychev(data, obj.interval)
	
	# Calculate chebychev polynomials
	T = calc_chebychevpoly(ν, obj.n)
	
	# Return fitted values
	return T * obj.θ
end #PREDICT.MYCHEBYCHEV

# Utility functions for calculation of myChebychev

function translate_from_chebychev(ν, interval)
    nobs, D = size(ν)
    Z = fill(0.0, nobs, D)
    for d in 1:D
        Z[:, d] = (ν[:, d] .+ 1) * 
            (interval[d, 2] - interval[d, 1]) / 2 .+ interval[d, 1]
    end
    return Z
end

function translate_to_chebychev(X, interval)
    nobs, D = size(X)
    ν = fill(0.0, nobs, D)
    for d in 1:D
        ν[:, d] = 2(X[:, d] .- interval[d, 1]) / 
            (interval[d, 2] - interval[d, 1]) .- 1
    end
    return ν
end

function calc_chebychevpoly(ν, n)
    # Data parameters
    L, D =  size(ν)
    K = (n+1)^D
    
    # Calculate by row
    T = fill(0.0, L, K)
    for j in 1:L

        t = fill(0.0, n+1)
        t[1] = 1
        # Compute jth row of T
        T_j = 1
        for d in 1:D
            # Compute Chebychev polynomial
            ν_jd = ν[j, d]
            t[2] = ν_jd
            for k in 2:n
                t[k+1] = 2ν_jd * t[k] - t[k-1]
            end
            # Apply kronecker product
            T_j = kron(t, T_j)
        end
        # Add to T
        T[j, :] = T_j
    end
    return T
end