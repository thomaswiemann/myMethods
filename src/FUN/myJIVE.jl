# Define myJIVE object
struct myJIVE
    coef::Array{Float64}
    y::Array{Float64} # response
    D::Array{Float64} # endogeneous variables
    Z::Array{Float64} # instruments
	X::Array{Float64} # features
	FS::Array{Float64} # first stage coefficients
    
    # Define constructor function
    function myJIVE(y, D, Z, X = ones(length(y)))
		# Data parameters
		N = length(y)
		
        # Define data matrices
        Z1 = hcat(X, Z)
        X1 = hcat(D, X)
        
        # Calculate first stage
        invZZ = inv(Z1' * Z1)
        DZ = X1' * Z1
        FS = invZZ * DZ'

        # Calculate leverage terms
        h = Array{Float64,1}(undef, N)
        ZinvZZ = Z1*invZZ
        for i in 1:N
            h[i] = ZinvZZ[i,:]' * Z1[i,:]
        end

        # Calculate LOO X
        loo_X = (Z1*FS .- h .* X1) ./ (1 .- h)

        # Calculate JIVE second stage
        coef = (loo_X'*X1)\(loo_X'*y) 
        
        # Return output
        new(coef, y, D, Z, X, FS)
    end #MYJIVE
end #MYJIVE

# Functions for objects of type myJIVE
## Coefficient function for myJIVE object
function coef(fit::myJIVE)
    return fit.coef
end #COEF.myJIVE

## Prediction function for myJIVE object
function predict(fit::myJIVE; data=nothing)
    # Return predicted values
    if isnothing(data)
		# Define data matrices
		X = hcat(fit.D, fit.X)
        return X * fit.coef
    else
        return data * fit.coef
    end
end #PREDICT.myJIVE

## Inference function for myJIVE object
# to do: clustered se
function inference(fit::myJIVE; heteroskedastic::Bool=false, cluster=nothing,
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

    # Covariance under homoskedasticity
    covar = sum(u.^2).*PZZPinv ./ (N-Kz-1)
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
end #INFERENCE.myJIVE