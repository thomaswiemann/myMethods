# Define myKDE object
# to do: set type of y!
struct myKDE
    y::Array{Float64,1} # response
	_y # values at which to calculate KDE
	h::Float64 # bandwidth
	kernel # kernel function
    
    # Define constructor function
    function myKDE(y::Array{Float64,1}; 
		_y = quantile(y[:], collect(1:10)./10),
		h::Float64=0.5, kernel = "Gaussian")	
    
    # Organize and return output
    new(y, _y, h, kernel)
    end #MYKDE
    
end #MYKDE

# Functions for objects of type myKDE
## prediction function for myKDE object
function predict(fit::myKDE, _y=fit._y)
    # Data parameters
    N_y = length(_y)
       
    # For each value in _x, calculate density
    den_mat = Array{Float64, 2}(undef, N_y, 1)
    idx = 0
    for yi in _y
        idx = idx + 1
        # Calculate distance to yi in the data normalized by the bandwidth
        u = (yi .- fit.y) ./ fit.h
        # Calculate kernel weights
        w = get_kw(u, fit.kernel)
        # Calculate density
        den_mat[idx] = mean(w) / fit.h
    end
    
    # Return kernel density estimate
    return den_mat
end #PREDICT.MYKDE