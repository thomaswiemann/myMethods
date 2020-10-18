# Define bootstrap function
# to do: optimize bootstrapping function arguments
function mybootstrap(FUN, y, B::Int, args...; 
        id=nothing, data_args=nothing, 
        get = nothing, red = nothing,
        key_args...)
		
    # Check for block bootstrap
	if(isnothing(id))
		id = collect(1:length(y))
	end
    u_id = unique(id)

    # Run bootstrap instances
    res = Array{Any}(nothing, B)
    for b in 1:B
		# Draw bootstrap sample
		boot_id = sample(u_id, length(u_id), replace = true)
		boot_sample = reduce(vcat, map(x -> findall(id.==x), boot_id))
		
        # Collect bootstrap samples
        y_b = y[boot_sample,:]
        ## Check whether additional data arguments have been passed
        args_b = args
        if !isnothing(data_args)
            args_b = collect(args_b) # convert to array for modification
            for k in data_args # uff...
				# Ensure that type of input remains unchanged
				if typeof(args_b[k]).== Array{Float64,2} || 
				typeof(args_b[k]).== Array{Int64,2}
					args_b[k] = args_b[k][boot_sample,:] 
				elseif typeof(args_b[k]).== Array{Float64,1} || 
				typeof(args_b[k]).== Array{Int64,1}
					args_b[k] = args_b[k][boot_sample]
				end					
            end
            args_b = tuple(args_b[:]...)# convert back to tuple
        end
        
        # Calculate and allocate results
        if !isnothing(get)
            res[b] = [get(FUN(y_b, args_b...; key_args...))]
        else 
            res[b] = [FUN(y_b, args_b...; key_args...)]
        end
    end
    
    # Organize and return output
    if !isnothing(red)
            res = reduce(red, res)
    end
    return res
end #MYBOOTSTRAP

# Define parallel bootstrap function
## Issues: 1. myLLR in mybootstrapPAR issues
function mybootstrapPAR(FUN, y, B::Int, args...; 
        id=nothing, data_args=nothing, 
        get = nothing, red = nothing,
		dynamic=false,
        key_args...)
    
	# Run parallel bootstrap procedure
	if !dynamic # w/o dynamic job scheduling
		res = @distributed (vcat) for b in 1:B
			mybootstrap(FUN, y, 1, args...; id=id, data_args=data_args, 
								get=get, red=nothing, key_args...)
		end
	else # w/ dynamic job scheduling	
		res = Vector{Any}(undef, B)
		@sync begin
			for p in workers()
				@async begin
					for b in 1:B
						res[b] = remotecall_fetch(mybootstrap, p, 
							FUN, y, 1, args...; id=id, data_args=data_args, 
							get=get, red=nothing, key_args...)
					end
				end
			end
		end
	end
    
    # Organize and return output
    if !isnothing(red)
            res = reduce(red, res)
    end
    return res
end #MYBOOTSTRAPPAR

# Helper function to reduce bootstrap output to array
function reduce_boot(boot_res)
    # Data parameters
    B = length(boot_res)
    N, K = size(reduce(hcat,boot_res[1]))
    if(N == 1)
        clean_boot_res = Array{Float64,2}(undef, K,B)
        for b in 1:B
        clean_boot_res[:,b] = reduce(hcat,boot_res[b])
        end
    else
        # Create three dimensional array and fill with results
        clean_boot_res = Array{Float64,3}(undef, N,K,B)
        for b in 1:B
            clean_boot_res[:,:,b] = reduce(hcat,boot_res[b])
        end
    end
    # Return results
    return clean_boot_res
end