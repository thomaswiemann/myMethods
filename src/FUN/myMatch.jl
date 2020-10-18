# Define myMatch object
struct myMatch
	ATT # ATT
	ATU # ATU
	ATE # ATE
	y # response
    D # binary treatment indicator
    x # matching variable
    K::Int64 # number of neighbours
    replacement::Bool # KNN with or without replacement
	S # weight matrix for KNN
	
	# Define constructor object
	function myMatch(y, D, x; 
		K::Int64=1, replacement::Bool=true, metric="Mahalanobis", S=nothing,
		obj = ["ATT", "ATU", "ATE"])
		
		# Data parameters
        N, nCol = size(x)
        nD1 = sum(D)
		
		# Check whether Cov_x should be calulated
		if nCol > 1 && isnothing(S) && 
			(metric == "Mahalanobis" || metric == "InvVariance")
			S = myCov(x)
			if metric == "InvVariance"
				S = diagm(S[diagind(S)])
			end
		end

        # Find subsamples
        isD0 = findall(D.==0)
        isD1 = setdiff(collect(1:N), isD0)
		
		# Calculate ATT
		ATT=nothing
		if any(obj.== "ATT") || any(obj.== "ATE")
			po_y0 = myKNN(y[isD0], x[isD0,:], x[isD1,:]; 
						K = K, replacement = replacement, metric=metric, S=S)
			ATT = mean(y[isD1] .- po_y0.fitted)
		end
		# Calculate ATU
		ATU=nothing
		if any(obj.== "ATU") || any(obj.== "ATE")
			po_y1 = myKNN(y[isD1], x[isD1,:], x[isD0,:]; 
						K = K, replacement = replacement, metric=metric, S=S)
			ATU = mean(po_y1.fitted .- y[isD0])
		end
		# Calculate ATE
		ATE=nothing
		if any(obj.=="ATE")
			ATE = ATT*(nD1/N) + ATU*(1-nD1/N)
		end
		
		# Organize and return output
		new(ATT, ATU, ATE, y, D, x, K, replacement, S)
	end#MYMATCH
end #MYMATCH

# Functions for objects of type myLS
## Coefficient function for myLS object
function coef(fit::myMatch, obj = "ATT")
	if obj == "ATT"
		return fit.ATT
	elseif obj == "ATU"
		return fit.ATU
	else 
		return fit.ATE
	end
end #COEF.MYLS