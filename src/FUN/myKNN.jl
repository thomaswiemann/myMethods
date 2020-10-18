# Define myKNN object
#To do: allow x to be multivariate; define mahalianobis distance
struct myKNN
    fitted # predicted values
	knn # matrix containing indices of k nearest neighbours
    y # response
    x # matching variable(s)
	_x # values to predict for
    K::Int64 # number of neighbours
    replacement::Bool # KNN with or without replacement
	metric # metric for multivariate matching
	
	# Define constructor function
    function myKNN(y, x, _x; K = 1, replacement = true, metric="Mahalanobis",
		S=nothing)
        # Data parameters
        N,nCol = size(x)
        n_x = size(_x,1)

        # Calculate the matched differences
        fitted = Array{Float64,2}(undef, n_x,1); 
		knn = Array{Int64,2}(undef, n_x, K); 
        is_out = fill(false, N); # for w/o replacement
        for idx in 1:n_x
            u = myDist(x[.!is_out,:],_x[idx,:], metric=metric, S=S) 
            knn[idx,:] = sortperm(u)[1:K] # K nearest neighbours
            fitted[idx] = mean(y[.!is_out][knn[idx,:]])
            #if(!replacement) 
            #    is_out[knn_idx] .= true
            #end
        end 
        
        # Organize and return output
        new(fitted, knn, y, x, _x, K, replacement, metric)
    end #MYKNN
end #MYKNN