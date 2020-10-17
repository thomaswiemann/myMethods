# Define mySieve object
struct mySieve
    coef # coefficients
    y # response
    X # matrix of regressors  
    basis # type of polynominal basis
    K # degree of sieve
	knots # knots for splines
    
    # Define constructor function
    function mySieve(y, x, control=nothing; 
            basis="Bernstein", K=3, knots = nothing)
		# Data parameters
		N = length(y)
        
        # Define matrix of regressors
        ## Calculate from basis
        X = get_basis(x, basis, K, knots)
        
        ## Check whether add. variables are included
        if !isnothing(control)
            X = hcat(X, control)
        end
        
        # Estimate sieve regression
        fit_sieve = myLS(y, X)
        coef = fit_sieve.coef
        
        # Organize and return output
        new(coef, y, X, basis, K, knots)
    end #MYSIEVE
    
end #MYSIEVE

# Functions for objects of type mySieve
## Coefficient function for mySieve object
function coef(fit::mySieve)
    return fit.coef
end #COEF.MYSIEVE

## Prediction function for mySieve object
## To do: create new regressors matrix from data!
function predict(fit::mySieve; x=nothing, control=nothing)
    # Return fitted values
    if isnothing(x)
		# Calculate and return in fitted values
        return fit.X * fit.coef
    else
		# Build new matrix of regressors
		X = get_basis(x, fit.basis, fit.K, fit.knots)
		## Check whether add. variables are included
        if !isnothing(control)
            X = hcat(X, control)
        end
		# Calculate and return fitted values
        return X * fit.coef
    end
end #PREDICT.MYSIEVE

# Internal functions (not exported)
## Internal function to construct matrix of regressors from basis
function get_basis(x, basis, K, knots)
	if basis == "Bernstein"
        X = [binomial(K, k).*(x.^k).*(1 .-x).^(K-k) for k in 0:K]
		X = reduce(hcat,X)
    elseif basis == "Monomial"
        X = [x.^k for k in 0:K]
		X = reduce(hcat,X)
	elseif basis == "LSplines"
		# Check whether sufficient knots are provided
		if isnothing(knots)
			#println("Warning: Knots defined as quantiles of x.")
			q_knots = collect(1:(K))./(K+1)
			knots = quantile(x[:], q_knots)
		else
			K = length(knots)
		end
		# Calculate splines
		X = [(x.>knots[k]).*(x.-knots[k]) for k in 1:K] # not very efficient...
		# Add constant and linear term
		X = hcat(fill(1,length(x)), x, reduce(hcat,X))
    end
	# Return matrix of regressors
	return X
end #GET_BASIS