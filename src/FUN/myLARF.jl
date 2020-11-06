# Define myLARF object
struct myLARF
    coef::Array{Float64}
    y::Array{Float64} # response
	D::Array{Float64} # endogeneous variables
	Z::Array{Float64} # instruments
    x::Array{Float64} # Sieve running variable
    X # features
    pX # features for sieve regression
    tau0::Array{Float64} # instrument probability
    kp::Array{Float64} # kappa
	K # sieve order
    
    function myLARF(y, D, Z, x; X = nothing, X2 = X,
        K = 1)
        # First stage sieve regression
        fit_sieve = mySieve(Z, x, X2; 
                basis="Monomial", K=K);
        tau0 = predict(fit_sieve);
        pX = fit_sieve.X;

        # Second stage WLS
        kp = @. 1 - D*(1-Z)/(1-tau0) - (1-D)*Z/tau0;
        fit = myLS(y, hcat(D, X), kp)
        
        # Return fit
        new(fit.coef, y, D, Z, x, X, pX, tau0, kp, K)
    end#MYLARF
end#MYLARF

# Functions for objects of type myLARF
## Coefficient function for myLARF object
function coef(fit::myLARF)
    return fit.coef
end #COEF.MYLARF

## Prediction function for myLARF object
function predict(fit::myLARF; data=nothing)
    # Return predicted values
    if isnothing(data)
		# Combined D and X for second stage regression
		X1 = hcat(fit.D, fit.X)
        return X1 * fit.coef
    else
        return data * fit.coef
    end
end #PREDICT.MYLARF

## Inference function for myLARF object
function inference(fit::myLARF; print_df::Bool=true)
    # Combined D and X for second stage regression
    X1 = hcat(fit.D, fit.X)

    # Data parameters
    N = length(fit.y)
    nX = size(X1, 2) # second stage parameters
    npX = size(fit.pX, 2) # first stage parameters

    # Calculate nu, residual u, and the gradient
    nu = @. fit.Z*(1-fit.D)/(fit.tau0^2) - fit.D*(1-fit.Z)/((1-fit.tau0)^2)
    dg = (fit.y - X1 * fit.coef) .* X1

    # Calculate delta
    pXXp = fit.pX' * fit.pX
    delta = fit.pX * inv(pXXp) * fit.pX' * (dg .* nu)

    # Calculate inner variance term and weighted inverse hessian   
    kgd = (dg .* fit.kp) + delta .* (fit.Z - fit.tau0)
    kgddgk = kgd' * kgd
    inv_M = inv((X1 .* fit.kp)' * X1) 

    # Calculate covariance matrix and extract standard errors
    covar = (inv_M * kgddgk * inv_M) 
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
end#INFERENCE.MYLARF