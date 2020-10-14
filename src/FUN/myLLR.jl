# Define myLLR object
# to do: set type of x!
struct myLLR
    y::Array{Float64,2} # response
    x # running variable
    K::Int64 # degree of local linear regression
	h::Float64 # bandwidth
	control # variables included in LLR
	kernel # kernel function
    
    # Define constructor function
	# to do: add mySilver function as default value!
    function myLLR(y::Array{Float64,2}, x; K::Int64=0,
		h::Float64=0.5, control = nothing, kernel = "Epanechnikov")	
    
    # Organize and return output
    new(y, x, K, h, control, kernel)
    end #MYLLR
    
end #MYLLR

# Functions for objects of type myLLR
## Coefficient function for myLLR object
function coef(fit::myLLR, x=quantile(fit.x, collect(1:10)./10))
    # Data parameters
    N_x = length(x)
    
    # Check whether additional variables are included 
    with_control = !isnothing(fit.control)
    if with_control
        dim_coef = size(control,2) + fit.K + 1
    else
        dim_coef = fit.K + 1
    end
    
    # For each value in x, calculate local coefficients
    coef_mat = Array{Float64, 2}(undef, N_x, dim_coef)
    idx = 0
    for xi in x
        idx = idx + 1
        # Calculate distance to xi in the data normalized by the bandwidth
        u = (xi .- fit.x) ./ fit.h
        # Calculate kernel weights
        w = get_kw(u, fit.kernel)
        # Create regressor matrix
        X = reduce(hcat, [u.^d for d in 0:fit.K])
        if !isnothing(fit.control)
            X = hcat(X, fit.control)
        end
        # Calculate WLS fit
        coef_mat[idx,:] = myLS(fit.y, X, w).coef
    end
    
    # Return local coefficients
    return coef_mat
end #COEF.MYLLR

## Parallel coefficient function for myLLR object
function coefPAR(fit::myLLR; 
        x=quantile(fit.x, collect(1:10)./10),
        dynamic=false,)
    # Data parameters
    N_x = length(x)
    # Check whether additional variables are included 
    with_control = !isnothing(fit.control)
    if with_control
        dim_coef = size(control,2) + fit.K + 1
    else
        dim_coef = fit.K + 1
    end
    
    # Run LLR in parallel
    if !dynamic # w/o dynamic job scheduling
        coef_mat = @distributed (hcat) for idx in 1:length(x)
            coef(fit, x[idx])'
        end
    else # w/ dynamic job scheduling
        coef_mat = Array{Float64, 2}(undef, N_x, dim_coef)
        @sync begin
            for p in workers()
                @async begin
                    for idx in 1:length(x)
                        res[idx,:] = remotecall_fetch(coef, p, 
                        fit, x[idx])'
                    end
                end
            end
        end
    end
    
    # Return local coefficients
    return coef_mat
end #COEFPAR.MYLLR
    
## Internal function to obtain kernel weights    
### Note: Scales are omitted b/c weights are rescaled to unity in sample
function get_kw(u, kernel)
    # Calculate kernel weights
    w = fill(0.0, length(u))
    if kernel == "Epanechnikov"
        @. w = (1 - u^2) * (abs(u) < 1)
    elseif kernel == "Uniform"
        @. w = (abs(u) < 1)
    elseif kernel == "Biweight"
        @. w = (1-u^2)^2
    elseif kernel == "Gaussian"
        w = pdf(Normal(), u.^2)
    else
        println("Specified kernel not yet implemented.")
    end
    # Normalize and return weights
    return w./sum(w)
end # GET_KW