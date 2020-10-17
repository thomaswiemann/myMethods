struct myMatch
	ATT # ATT
	ATU # ATU
	ATE # ATE
	y # response
    D # binary treatment indicator
    x # matching variable
    K::Int64 # number of neighbours
    replacement::Bool # KNN with or without replacement
	
	# Define constructor object
	function myMatch(y, D, x; 
		K::Int64=1, replacement::Bool=true, obj = ["ATT", "ATU", "ATE"])
		
		# Data parameters
        N = length(y)
        nD1 = sum(D)

        # Find subsamples
        isD0 = findall(D.==0)
        isD1 = setdiff(collect(1:N), isD0)
		
		# Calculate ATT
		ATT=nothing
		if any(obj.== "ATT") || any(obj.== "ATE")
			po_y0 = myKNN(y[isD0], x[isD0], _x[isD1]; 
						K = K, replacement = replacement)
			ATT = mean(y[isD1] .- po_y0.fitted)
		end
		# Calculate ATU
		ATU=nothing
		if any(obj.== "ATU") || any(obj.== "ATE")
			po_y1 = myKNN(y[isD1], x[isD1], _x[isD0]; 
						K = K, replacement = replacement)
			ATT = mean(po_y1.fitted .- y[isD0])
		end
		# Calculate ATE
		ATE=nothing
		if any(a.=="ATE")
			ATE = ATT*(nD1/N) + ATU*(1-nD1/N)
		end
		
		# Organize and return output
		new(ATT, ATU, ATE, y, D, x, K, replacement)
	end#MYMATCHING
end #MYMATCHING