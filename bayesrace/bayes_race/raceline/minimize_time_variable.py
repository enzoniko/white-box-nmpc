""" Calculate minimum lap time and optimal speed profile on a fixed path with variable friction.
    Friction circle model is used for vehicle dynamics.
    
    Extension of minimize_time.py to support position-dependent friction maps.

    Implementation of ``Minimum-time speed optimisation over a fixed path`` by Lipp and Boyd (2014)
    https://web.stanford.edu/~boyd/papers/pdf/speed_opt.pdf
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import cvxpy as cv
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

from bayes_race.utils import odeintRK6
from bayes_race.params import ORCA, F110
from bayes_race.tracks import ETHZ, ETHZMobil, UCB


def define_params(mass, lf, lr, friction_func=None, friction_args=None):
    """ params derived from vehicle config
    
    Args:
        mass: Vehicle mass [kg]
        lf: Distance from CG to front axle [m]
        lr: Distance from CG to rear axle [m]
        friction_func: Optional function f(x, y) -> μ (friction coefficient)
        friction_args: Optional dict of additional arguments for friction_func
    
    Returns:
        params: Dictionary of parameters
    """
    gravity = 9.81          # gravity
    Wr = lf/(lf+lr)         # percent of weight on rear tires 
    
    # Default friction if not provided (backward compatible)
    if friction_func is None:
        muf = 1.0
        Fmax = muf*mass*gravity
        Flongmax = Fmax*Wr
    else:
        # Friction will be computed per-segment in optimize()
        # Store function for later use
        Fmax = None  # Placeholder
        Flongmax = None  # Placeholder
    
    params = {}
    params['lf'] = lf
    params['lr'] = lr
    params['mass'] = mass
    params['Fmax'] = Fmax
    params['Flongmax'] = Flongmax
    params['Wr'] = Wr
    params['gravity'] = gravity
    params['friction_func'] = friction_func
    params['friction_args'] = friction_args if friction_args is not None else {}
    return params

def define_path(x, y, plot_results=True):
    """ calculate derivatives and double derivatives in path coordinate system
    """
    num_wpts = np.size(x)
    theta = np.linspace(0, 1, num_wpts)
    dtheta = 1/(num_wpts-1)

    S = np.array([x, y])
    S_middle = np.zeros([2,num_wpts-1])
    S_prime = np.zeros([2,num_wpts-1])
    S_dprime= np.zeros([2,num_wpts-1])

    for j in range(num_wpts-1):
        S_middle[:,j] = (S[:,j] + S[:,j+1])/2
        S_prime[:,j] = (S[:,j+1] - S[:,j])/dtheta
        if j==0:
            S_dprime[:,j] = (S[:,j]/2 - S[:,j+1] + S[:,j+2]/2)/(dtheta**2)
        elif j==1 or j==num_wpts-3:
            S_dprime[:,j] = (S[:,j-1] - S[:,j] - S[:,j+1] + S[:,j+2])/2/(dtheta**2)
        elif j==num_wpts-2:
            S_dprime[:,j] = (S[:,j-1]/2 - S[:,j] + S[:,j+1]/2)/(dtheta**2)
        else:
            S_dprime[:,j] = (- 5/48*S[:,j-2] + 13/16*S[:,j-1] - 17/24*S[:,j] - 17/24*S[:,j+1] + 13/16*S[:,j+2] - 5/48*S[:,j+3])/(dtheta**2)

    path = {
            'theta': theta, 
            'dtheta': dtheta, 
            'S': S, 
            'S_middle': S_middle, 
            'S_prime': S_prime, 
            'S_dprime': S_dprime,
            }
    return path

def dynamics(phi, params):
    """ dynamics (non-linear)
    """
    mass = params['mass']
    M = np.zeros((2,2))
    M[0,0] = mass
    M[1,1] = mass

    R = np.zeros((2,2))
    R[0,0] = np.cos(phi)
    R[0,1] = -np.sin(phi)
    R[1,0] = np.sin(phi)
    R[1,1] = np.cos(phi)
    
    C = np.zeros((2,2))
    d = np.zeros((2))
    return R, M, C, d

def dynamics_cvx(S_prime, S_dprime, params):
    """ dynamics (convexified)
    """
    phi = np.arctan2(S_prime[1],S_prime[0])
    R, M, C, d = dynamics(phi, params)
    C = np.dot(M, S_dprime) + np.dot(C, S_prime)
    M = np.dot(M, S_prime)
    return R, M, C, d

def diffequation(t, x, u, R, M, C, d):
    """ write as first order ode
    """
    x0dot = x[2:]
    x1dot = np.dot(np.linalg.inv(M), np.dot(R, u) - np.dot(C, x[2:]) - d)
    return np.concatenate([x0dot, x1dot], axis=0)

def plots(t, x, y, vxy, u, S, params):
    # position
    flg, ax = plt.subplots(1)
    plt.plot(S[0,:], S[1,:], '-b.', label='waypoints')
    plt.plot(x, y, 'r.', label='open-loop sim')
    plt.plot(S[0,0], S[1,0], 'g*', label='start')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis("equal")
    plt.title('global position')
    plt.legend(loc=0)

    # gg plot
    flg, ax = plt.subplots(1)
    # For variable friction, use average Fmax for visualization
    if isinstance(params.get('Fmax'), (list, np.ndarray)):
        Fmax_avg = np.mean(params['Fmax'])
    else:
        Fmax_avg = params['Fmax']
    c1 = friction_circle(Fmax_avg)
    plt.plot(u[1,:]/params['mass'], u[0,:]/params['mass'], '.')
    plt.plot(c1[0]/params['mass'], c1[1]/params['mass'], '-k', lw=1, alpha=0.5)
    plt.xlabel('lateral acceleration [m/s^2]')
    plt.ylabel('longitudinal acceleration [m/s^2]')
    plt.grid(True)
    plt.axis('equal')
    plt.title('operating points on GG-plot')

    # speed
    flg, ax = plt.subplots(1)
    plt.plot(t, vxy, label='speed abs')
    plt.title('absolute speed vs time')
    plt.xlabel('time [s]')
    plt.ylabel('speed [m/s]')
    plt.grid(True)
    plt.legend(loc=0)

    # control inputs
    flg, ax = plt.subplots(1)
    plt.plot(t[1:], u[0,:], '-', label='longitudinal')
    plt.plot(t[1:], u[1,:], '-', label='lateral')
    plt.title('force vs time')
    plt.xlabel('time [s]')
    plt.ylabel('force [N]')
    plt.grid(True)
    plt.legend(loc=0)
    plt.show()

def friction_circle(Fmax):
    t = np.linspace(0, 2*np.pi, num=100)
    x = Fmax*np.cos(t)
    y = Fmax*np.sin(t)
    return x, y

def get_time_vec(b, theta):
    num_wpts = theta.size
    dtheta = 1/(num_wpts-1)
    bsqrt = np.sqrt(b)
    dt = 2*dtheta/(bsqrt[0:num_wpts-1]+bsqrt[1:num_wpts])
    t = np.zeros([num_wpts])
    for j in range(1, num_wpts):
        t[j] = t[j-1] + dt[j-1]
    return t

def simulate(b, a, u, path, params, plot_results, print_updates, int_method ='rk6'):
    """ integrate using ode solver
    """
    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    num_wpts = theta.size

    # initialize position, velocity
    x, y = np.zeros([num_wpts]), np.zeros([num_wpts])
    x[0], y[0] = S[0,0], S[1,0]
    vx, vy = np.zeros([num_wpts]), np.zeros([num_wpts])

    # calculate time for each index
    t = get_time_vec(b, theta)

    # integrate
    if int_method == 'odeint':
        if print_updates:
            print('using Runge Kutta sixth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j],S_prime[0,j])
            R, M, C, d = dynamics(phi, params)
            odesol = odeint(diffequation, [x[j], y[j], vx[j], vy[j]], [t[j], t[j+1]], 
                            args=(u[:,j], R, M, C, d), tfirst=True)
            x[j+1], y[j+1], vx[j+1], vy[j+1] = odesol[-1,:]

    elif int_method == 'rk6':
        if print_updates:
            print('using Runge Kutta sixth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j],S_prime[0,j])
            R, M, C, d = dynamics(phi, params)
            odesol = odeintRK6(diffequation, [x[j], y[j], vx[j], vy[j]], [t[j], t[j+1]], 
                            args=(u[:,j], R, M, C, d))
            x[j+1], y[j+1], vx[j+1], vy[j+1] = odesol[-1,:]

    else:
        integrator = ode(diffequation).set_integrator('dopri5')
        if print_updates:
            print('using Runge Kutta fourth order integration')
        for j in range(num_wpts-1):
            phi = np.arctan2(S_prime[1,j],S_prime[0,j])
            R, M, C, d = dynamics(phi, params)
            integrator.set_initial_value([x[j], y[j], vx[j], vy[j]], t[j]).set_f_params(u[:,j], R, M, C, d)
            x[j+1], y[j+1], vx[j+1], vy[j+1] = integrator.integrate(t[j+1])

    vxy = np.sqrt(vx**2+vy**2)
    if plot_results:
        plots(t, x, y, vxy, u, S, params)
    
    return vxy, t[-1]

def optimize(path, params, plot_results, print_updates):
    """ main function to solve convex optimization with variable friction
    
    Key modification: Pre-computes friction at segment midpoints to maintain convexity.
    """
    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    S_middle = path['S_middle']  # Midpoints for friction evaluation
    num_wpts = theta.size
    
    # Pre-compute friction at each segment midpoint
    friction_func = params.get('friction_func', None)
    friction_args = params.get('friction_args', {})
    
    if friction_func is not None:
        # Evaluate friction at segment midpoints
        mu_segments = friction_func(S_middle[0, :], S_middle[1, :], **friction_args)
        # Ensure mu_segments is 1D array
        if not isinstance(mu_segments, np.ndarray):
            mu_segments = np.array(mu_segments)
        if mu_segments.ndim > 1:
            mu_segments = mu_segments.flatten()
        
        # Compute per-segment force limits
        gravity = params['gravity']
        mass = params['mass']
        Wr = params['Wr']
        Fmax_segments = mu_segments * mass * gravity
        Flongmax_segments = Fmax_segments * Wr
        
        if print_updates:
            print(f"Variable friction: min μ={mu_segments.min():.2f}, max μ={mu_segments.max():.2f}")
    else:
        # Constant friction (original behavior)
        Fmax_segments = np.ones(num_wpts - 1) * params['Fmax']
        Flongmax_segments = np.ones(num_wpts - 1) * params['Flongmax']
    
    # opt vars
    A = cv.Variable((num_wpts-1))
    B = cv.Variable((num_wpts))
    U = cv.Variable((2, num_wpts-1))

    cost = 0
    constr = []

    # no constr on A[0], U[:,0], defined on mid points
    constr += [B[0] == 0]

    for j in range(num_wpts-1):

        cost += 2*dtheta*cv.inv_pos(cv.power(B[j],0.5) + cv.power(B[j+1],0.5))

        R, M, C, d = dynamics_cvx(S_prime[:,j], S_dprime[:,j], params)
        constr += [R*U[:,j] == M*A[j] + C*((B[j] + B[j+1])/2) + d]
        constr += [B[j] >= 0]
        
        # Position-dependent friction constraint
        constr += [cv.norm(U[:,j],2) <= Fmax_segments[j]]
        constr += [U[0,j] <= Flongmax_segments[j]]
        constr += [B[j+1] - B[j] == 2*A[j]*dtheta]
        
    problem = cv.Problem(cv.Minimize(cost), constr)
    # Try ECOS first, fallback to SCS if it fails
    try:
        solution = problem.solve(solver=cv.ECOS, verbose=False)
    except Exception as e:
        print("   ECOS solver failed, trying SCS...")
        solution = problem.solve(solver=cv.SCS, verbose=False, max_iters=5000)

    B, A, U = B.value, A.value, U.value
    B = abs(B)
    vopt, topt = simulate(B, A, U, path, params, plot_results=plot_results, print_updates=print_updates)
    return B, A, U, vopt, topt

def solve(x, y, mass, lf, lr, friction_func=None, friction_args=None, plot_results=False, print_updates=False, **kwargs):
    """ call this wrapper function
    
    Args:
        x, y: Path waypoints
        mass, lf, lr: Vehicle parameters
        friction_func: Optional function f(x, y) -> μ
        friction_args: Optional arguments for friction_func
        plot_results: Whether to plot results
        print_updates: Whether to print progress
        **kwargs: Additional arguments (for backward compatibility)
    """
    path = define_path(x, y)
    params = define_params(mass, lf, lr, friction_func=friction_func, friction_args=friction_args)
    B, A, U, vopt, topt = optimize(
        path=path, 
        params=params, 
        plot_results=plot_results, 
        print_updates=print_updates,
        )
    return x, y, vopt, topt, U, B

def calcMinimumTime(x, y, friction_func=None, friction_args=None, **kwargs):
    """ wrapper function to return minimum time only
    """
    x, y, vopt, topt, U, B = solve(x, y, friction_func=friction_func, friction_args=friction_args, **kwargs)
    return topt

def calcMinimumTimeSpeed(x, y, friction_func=None, friction_args=None, **kwargs):
    """ wrapper function to return minimum time and speed profile only
    """
    x, y, vopt, topt, U, B = solve(x, y, friction_func=friction_func, friction_args=friction_args, **kwargs)
    return topt, vopt

def calcMinimumTimeSpeedInputs(x, y, friction_func=None, friction_args=None, **kwargs):
    """ wrapper function to return minimum time and speed profile only
    """
    x, y, vopt, topt, U, B = solve(x, y, friction_func=friction_func, friction_args=friction_args, **kwargs)
    path = define_path(x, y)
    return get_time_vec(B, path['theta']), vopt, U

def calcInputs(x, y, friction_func=None, friction_args=None, **kwargs):
    """ wrapper function to return control inputs only
    """
    x, y, vopt, topt, U, B = solve(x, y, friction_func=friction_func, friction_args=friction_args, **kwargs)
    return U   


if __name__ == "__main__":
    """ example how to use
    """
    from bayes_race.utils.friction import get_friction
    
    # define waypoints
    track = UCB()
    x, y = track.x_center, track.y_center

    # define vehicle params
    params = F110()

    # call the solver with variable friction
    start = time.time()
    topt = calcMinimumTime(x, y, friction_func=get_friction, **params)
    end = time.time()
    print("time to solve optimization: {:.2f}".format(end-start))
