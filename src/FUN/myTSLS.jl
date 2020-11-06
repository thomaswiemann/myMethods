# Define myTSLS object
struct myTSLS
    coef::Array{Float64}
    y::Array{Float64} # response
	D::Array{Float64} # endogeneous variables
	Z::Array{Float64} # instruments
    X::Array{Float64} # features
	FS::Array{Float64} # first stage coefficients
	
	# Define constructor function
	function myTSLS(y, D, Z; X = ones(length(y)))
		# Define data matrices
		Z1 = hcat(X, Z)
		X1 = hcat(D, X)
		
		# Calculate matrix products
		ZZ = Z1' * Z1
		DZ = X1' * Z1
		Zy = Z1' * y
		FS = inv(ZZ) * DZ'
		
		# Calculate TSLS coefficient
		coef = (DZ * FS)' \ (FS' * Zy)
		
		# Return output
		new(coef, y, D, Z, X, FS)
	end #MYTSLS
end#MYTSLS

# Functions for objects of type myTSLS
## Coefficient function for myTSLS object
function coef(fit::myTSLS)
    return fit.coef
end #COEF.myTSLS

## Prediction function for myTSLS object
function predict(fit::myTSLS; data=nothing)
    # Return predicted values
    if isnothing(data)
		# Define data matrices
		X = hcat(fit.D, fit.X)
        return X * fit.coef
    else
        return data * fit.coef
    end
end #PREDICT.myTSLS

## Inference function for myTSLS object
# to do: clustered se
function inference(fit::myTSLS; heteroskedastic::Bool=false, cluster=nothing,
    print_df::Bool=true)
    # Obtain data parameters
    N = length(fit.y)
	
	# Define sample matrices
	Z1 = hcat(fit.X, fit.Z); Kz = size(Z1,2)
	X1 = hcat(fit.D, fit.X); Kx = size(X1,2)
	u = fit.y - predict(fit) # residuals  
	
	# Calculate matrix products
	PZ = Z1 * fit.FS
	PZZPinv = inv(PZ' * PZ)

	# Calculate covariance matrix
	if !heteroskedastic
		# Covariance under homoskedasticity
		covar = sum(u.^2).*PZZPinv ./ (N-Kz)
	else
		# Covariance under heteroskedasticity
		PZuuZP = ((PZ .* (u.^2))' * PZ) .* (N/(N-Kz))
		covar = PZZPinv * PZuuZP * PZZPinv
	end
	
	# Get standard errors
    se = sqrt.(covar[diagind(covar)])

    # Calculate t-statistic and p-values
    t_stat = fit.coef./se
    p_val = 2*cdf(Normal(), -abs.(t_stat))
    
    # Define output
    output = (coef=fit.coef, se=se, t=t_stat, p=p_val)
    # Print estimates
    if print_df
        out_df = convert(DataFrame, hcat(fit.coef, se, t_stat, p_val))
        rename!(out_df, ["coef", "se", "t-stat", "p-val"])
        display(out_df)
    end
    # Organize and return output
    output = (coef=fit.coef, se=se, t=t_stat, p=p_val)
    return output
end #INFERENCE.myTSLS