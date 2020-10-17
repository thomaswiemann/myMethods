# Additional smaller functions

# Function for empirical covariance matrix b/w rnd variables
function myCov(X)
	EXX = X'*X
	EX =  mapslices(mean, X, dims=1) 
	Cov = EXX - EX'*EX
	return Cov
end #MYCOV

# Function for Mahalanobis distance
function myMahalanobis(x, y, S)
	return sqrt((x-y)'*inv(S)*(x-y))
end #MYMAHALANOBIS