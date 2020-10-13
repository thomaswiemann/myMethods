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