# Define myLS object
struct myLS
    coef::Array{Float64,2}
    y::Array{Float64,1} # response
    X::Array{Float64,2} # features
    w # weights
    
    # Define constructor function
    function myLS(y::Array{Float64,1}, X::Array{Float64,2}, w=nothing;
	constant::Bool=false)
    # Check whether constant should be added to X
    if constant
        X = hcat(ones(size(X)[1],1), X)
    end
	
    # Calculate LS or WLS
    if isnothing(w)
        coef = (X'*X)\(X'*y) 
    else
        Xw = X.*w
        coef = (Xw'*X)\(Xw'*y)
    end
    
    # Organize and return output
    new(coef, y, X, w)
    end #MYLS
    
end #MYLS

# Functions for objects of type myLS
# Coefficient function for myLS object
function coef(fit::myLS)
    return fit.coef
end #COEF.MYLS

## Prediction function for myLS object
function predict(fit::myLS; data=nothing)
    # Return predicted values
    if isnothing(data)
        return fit.X * fit.coef
    else
        return data * fit.coef
    end
end #PREDICT.MYLS

## Inference function for myLS object
## To do: implement clustered standard errors
function inference(fit::myLS; heteroskedastic::Bool=false, 
    print_df::Bool=true)
    # Obtain data parameters
    N = length(fit.y)
    K = size(fit.X)[2]
    
    # Calculate covariance matrix and standard errors
    u = fit.y - predict(fit) # residuals
    if isnothing(fit.w)
        # Covariance for LS 
        XX_inv = inv(fit.X'*fit.X)
        if !heteroskedastic
            covar = sum(u.^2) * XX_inv
			covar = covar .* (1/(N-K)) # dof adjustment
        else
            covar = XX_inv * ((fit.X .* (u.^2))' * fit.X) * XX_inv 
			covar = covar .* (N/(N-K)) # dof adjustment
        end
    else 
        # Covariance for WLS
        # Question: different dof adjustment for WLS?
        Xw = fit.X.*fit.w
        XwX_inv = inv(Xw'*fit.X)
        if !heteroskedastic
            covar = sum(fit.w.*(u.^2)) * XwX_inv .* (1/(N-K)) 
			covar = covar .* (1/(N-K)) # dof adjustment
        else
            covar = XwX_inv * ((fit.X .* ((fit.w.*u).^2))' * fit.X) * XwX_inv 
			covar = covar .* (N/(N-K)) # dof adjustment
        end
    end
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
end #INFERENCE.MYLS