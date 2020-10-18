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
	metric # metric for distance matrix
	S # weight matrix for KNN
	knn_ATT # matrix with neighbour indices
	knn_ATU # matrix with neighbour indices
	fitted_ATT
	fitted_ATU
	
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
		ATT=knn_ATT=fitted_ATT=nothing
		if any(obj.== "ATT") || any(obj.== "ATE")
			po_y0 = myKNN(y[isD0], x[isD0,:], x[isD1,:]; 
						K = K, replacement = replacement, metric=metric, S=S)
			ATT = mean(y[isD1] .- po_y0.fitted)
			knn_ATT = po_y0.knn
			fitted_ATT = po_y0.fitted
		end
		# Calculate ATU
		ATU=knn_ATU=fitted_ATU=nothing
		if any(obj.== "ATU") || any(obj.== "ATE")
			po_y1 = myKNN(y[isD1], x[isD1,:], x[isD0,:]; 
						K = K, replacement = replacement, metric=metric, S=S)
			ATU = mean(po_y1.fitted .- y[isD0])
			knn_ATU = po_y1.knn
			fitted_ATU = po_y1.fitted
		end
		# Calculate ATE
		ATE=nothing
		if any(obj.=="ATE")
			ATE = ATT*(nD1/N) + ATU*(1-nD1/N)
		end
		
		# Organize and return output
		new(ATT, ATU, ATE, y, D, x, K, replacement, metric, S, 
			knn_ATT, knn_ATU, fitted_ATT, fitted_ATU)
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

## Inference function for myMatch object
# To do: implement homoskedastic errors
# To do: implement errors for ATU and ATE
function inference(fit::myMatch; obj = "ATT", K = fit.K)
	# Data parameters
	N, nCol = size(fit.x)
	nD1 = sum(fit.D)
	
	# Find subsamples
    isD0 = findall(fit.D.==0)
    isD1 = setdiff(collect(1:N), isD0)
	
	if obj == "ATT"
		# Calculate Vtx
		VTx = mean((fit.y[isD1] .- fit.fitted_ATT .- fit.ATT).^2)
		
		# Calculate VEt -- to do: make more concise!
		## For all treated, find closest other treated and take difference
		sigs = Array{Float64,1}(undef, N)
        x_idx = Array{Float64,2}(undef, 1, nCol)
		#for idx in isD1
		#	isD1_idx = setdiff(isD1, idx)
        #    x_idx[1,:] = fit.x[idx,:]
		#	diff_idx = fit.y[idx] - myKNN(fit.y[isD1_idx], fit.x[isD1_idx,:], x_idx,
		#		K=K, replacement=fit.replacement, metric=fit.metric, S=fit.S).fitted[1]
        #    sigs[idx] = ((diff_idx)^2)*(K/(K+1))
		#end
        ## For all untreated
		for idx in isD0
			isD0_idx = setdiff(isD0, idx)
            x_idx[1,:] = fit.x[idx,:]
			diff_idx = fit.y[idx] - myKNN(fit.y[isD0_idx], fit.x[isD0_idx,:], x_idx,
				K=K, replacement=fit.replacement, metric=fit.metric, S=fit.S).fitted[1]
            sigs[idx] = ((diff_idx)^2)*(K/(K+1))
		end
        ## For all untreated, determine how often they are neighbours
        Km = Array{Float64,1}(undef, N)
        for idx in isD0
            Km[idx] = sum(fit.knn_ATT.==idx)
        end
        ## Combine terms to obtain VEt
        Km2 = Km.*(Km.-1)./(fit.K^2)
        VEt = sum(sigs[isD0].*Km2[isD0])/nD1
        
        # Calculate and return variance of ATT
        return (VTx+VEt)/N
	end
end #INFERENCE.MYMATCH





























