# Define function for the Anderson Rubin test
function myAndersonRubin(beta, y, D, Z; X = ones(length(y)), 
        heteroskedastic = false, calc_conf_set = false, plevel = 0.05, ueps = 1e-6)
    # Data parmeters
    nZ = size(Z, 2)
    # Residualize y, D, Z wrt X
    if !isnothing(X)
        y = y - predict(myLS(y, X))
        D = D - predict(myLS(D, X))
        for j in 1:nZ
            Z[:, j] = Z[:, j] - predict(myLS(Z[:, j], X))
        end
    end
    Z = Z[:,:] # convert to Array{Float64,2}

    # Project Y - D*beta onto Z
    fit_LS = myLS(y-D*beta, Z)

    # Calculate Anderson Rubin test statistic
    if nZ == 1
        # For1 instrument, calculate t test statistic
        inf_LS = inference(fit_LS, heteroskedastic=true, print_df=false)
        AR = inf_LS.t[1]^2
        p = inf_LS.p[1]
        se =  inf_LS.se[1]
    else
        # For more than 1 instrument, calculate F test statistic directly
    end

# Calculate confidence interval via test reversion
conf_set = zeros(1,2)
    if calc_conf_set
        for ub in [true, false]

            # Initial guess
            coef_beta = [beta, beta]
            reject_beta = [p<plevel, p<plevel]

        while true
            # Conditional exit statement
            diff_cond = ((coef_beta[2]-coef_beta[1]) < ueps) 
            if ub && (diff_cond && !reject_beta[1] && reject_beta[2])
                break
            elseif !ub && (diff_cond && reject_beta[1] && !reject_beta[2])
                break
            end

            # new proposal for bounds
            runif = rand(1)[1]
            newl_beta = coef_beta[1] + (coef_beta[2]-coef_beta[1])*runif
            newu_beta = coef_beta[2] - (coef_beta[2]-coef_beta[1])*(1-runif)

            # find larger lower bound
           while true
                # Test new proposal
                isreject = (myAndersonRubin(newl_beta, y, D, Z; X = X, 
                            heteroskedastic = heteroskedastic).p < plevel)
                # Check whether new proposal satisfies requirements
                if (ub && !isreject) || (!ub && isreject)
                  break
                end
                # Adjust new proposal
                newl_beta = newl_beta .- rand(1)[1]*se
            end
            # Accept or reject new upper bound
            if newl_beta > coef_beta[1] || (ub && reject_beta[1]) || (!ub && !reject_beta[1])
                coef_beta[1] = newl_beta
            end

            # find smaller upper bound
           while true
                # Test new proposal
                isreject = (myAndersonRubin(newu_beta, y, D, Z; X = X, 
                            heteroskedastic = heteroskedastic).p < plevel)
                # Check whether new proposal satisfies requirements
                if (ub && isreject) || (!ub && !isreject)
                  break
                end
                # Adjust new proposal
                newu_beta = newu_beta .+ rand(1)[1]*se
            end
            # Accept or reject new upper bound
            if newu_beta < coef_beta[2] || (ub && !reject_beta[2]) || (!ub && reject_beta[2])
                coef_beta[2] = newu_beta
            end

            # Update rejection step
            reject_beta = [!ub, ub]    
        end

        # Return rejected value to provide conservative bounds on confidence set
        conf_set[1,2^ub] = coef_beta[2^ub]
        end
    end

    # Organize and return output
    return (AR=AR, p=p, se=se, conf_set = conf_set)
end