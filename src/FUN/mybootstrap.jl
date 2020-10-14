# Define bootstrap function
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
            for k in data_args
                args_b[k] = args_b[k][boot_sample,:] # uff...
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