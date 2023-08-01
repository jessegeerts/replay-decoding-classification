import numpy as np

  
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        func_takes_multiple=False, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, 
        maxiter=100, minstep=1e-8, minfunc=1e-8, debug=False, info=True):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    func_takes_multiple : boolean
        If True, func has been designed to accept a matrix [len(lb) x swarmsize]
        and it will return a vector of length swarmsize. When False, func only
        accepts a vector of length len(lb).
        (Default: False)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    info : boolean
        If True, information statements will be displayed at the end,
        (Default: False)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    This is from https://github.com/tisimst/pyswarm/ on 02-Feb-2015.
    Vectorisation & additional tweaks by DM. Note this version has not been extensively tested.
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)[np.newaxis,:]
    ub = np.array(ub)[np.newaxis,:]
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Prepare main function for minimsing #####################################
    if func_takes_multiple:
        obj = lambda x: np.asarray(func(x.T, *args, **kwargs))
    else:
        obj = lambda x: np.array([func(xx, *args, **kwargs) for xx in x])
        
    # Prepare constrainet checking function ###################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            is_feasible = lambda x: True
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            is_feasible = lambda x: np.all((y(x, *args, **kwargs)>=0 for y in ieqcons))
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        is_feasible = lambda x: np.all(np.asarray(f_ieqcons(x, *args, **kwargs))>=0)
        
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    # Initialize the particle's position
    x = lb + x*(ub - lb)
    
    # Initialize the particle's best known position
    p = x

    # Calculate the objective's value at the current particle's
    fp = obj(p)

    # At the start, there may not be any feasible starting point, so just
    # give it a temporary "best" point since it's likely to change    
    g = p[0, :].copy()

    # If the current particle's position is better than the swarm's,
    # update the best swarm position    
    for i in range(S):
        if fp[i]<fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(D)*(vhigh - vlow)
       
    # Iterate until termination criterion met #################################
    for it in range(1,maxiter+1):
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        
        # Update the particles' velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)

        # Update the particles' positions, clipping lower and upper bounds
        x += v
        np.clip(x,lb,ub,out=x) 
        
        # Evaluate the function
        fx = obj(x)

        # update particles indiviual best values
        is_best = fx<fp
        if np.any(is_best):
            is_best[is_best] &= np.array([ is_feasible(xx) for xx in x[is_best,:] ])
            fp[is_best] = fx[is_best]
            p[is_best,:] = x[is_best,:]
        
        # is the current best of the best a new record
        fx_min_ind = np.argmin(fx)
        if fx[fx_min_ind] < fg:
            # (Can only get here if constraints are satisfied)
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))
            g_old = g
            fg_old = fg
            g = x[fx_min_ind, :].copy()
            fg = fx[fx_min_ind]

            stepsize = np.sqrt(np.sum((g-g_old)**2))
            if np.abs(fg - fg_old)<=minfunc:
                if info:
                    print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                return g, fg
            elif stepsize<=minstep:
                if info:
                    print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                return g, fg

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))


    if info:
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g) and info:
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg
