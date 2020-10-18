# Define myProbit object
struct myProbit
    coef # coefficient
    y # response
    X # regressors 
    
    # Define constructor function
    function myProbit(y, X; starting=nothing)
        # Data parameters
        N, nCol = size(X)

        # Check for starting parameters
        if isnothing(starting)
            starting = coef(myLS(y,X))
        end

        # Set up optimization
        func = TwiceDifferentiable(vars -> logl_probit(y, X, 
                vars, negative=true),starting; autodiff=:forward);
        # Run optimization
        opt = optimize(func, ones(nCol)./100)

        # Extract coefficients
        coefs = Optim.minimizer(opt)

        new(coefs, y, X)
    end
    
end

function predict(fit::myProbit)
    mu = fit.X * fit.coef
    return cdf.(Normal(0,1), mu)
end

function logl_probit(y, X, par; negative = false)
    # Calculate cond mean
    mu = X*par
    # Calculate (negative) logl
    logl = (-1)^negative .* sum(cdf_probit.(y,mu,clog=true))
    return logl
end

function cdf_probit(y, mu; clog=false)
    # Calculate density
    density = (1-y) + ((-1)^(1-y)) * cdf.(Normal(0, 1), mu)
    if clog
        density = log.(density)
    end
    # return density
    return density
end