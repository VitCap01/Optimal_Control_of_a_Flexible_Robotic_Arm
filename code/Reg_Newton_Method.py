import numpy as np
import dynamics as dyn
from dynamics import f
from cost import stagecost,termcost
from LQR import lqr_affine
from armijo import armijo_stepsize
import matplotlib.pyplot as plt

# Number of states
ns = dyn.ns
# Number of inputs
ni = dyn.ni

def traj_iteration_plot(x, u, xref, uref, maxiters, traj_show, dt, T):
    
    """
    This function visualizes the evolution of state and input trajectories over multiple iterations 
    of the Newton's method. It generates subplots for each state trajectory and the input trajectory 
    across time, comparing them with their respective reference curves at each iteration.

    Inputs:
    - x: 3D numpy array of state trajectories, shape (ns, T, maxiters).
        `ns` is the number of states, `T` is the time horizon, and `maxiters` is the number of iterations.
    - u: 3D numpy array of input trajectories, shape (1, T, maxiters).
        Represents the control input trajectory at each iteration.
    - xref: 2D numpy array of reference state curve, shape (ns, T).
        Represents the desired state curve over the time horizon.
    - uref: 2D numpy array of reference input curve, shape (1, T).
        Represents the desired input curve over the time horizon.
    - maxiters: Integer, the number of iterations to visualize.
    - traj_show: Boolean, if True, the function will plot the trajectories; otherwise, it does nothing.
    - dt: Float, it's the discretization step.
    - T: Integer, the number of discrete time steps in the time horizon.

    Outputs:
    - No explicit return value; the function displays plots of state and input trajectories for each iteration.
    """


    # Final time
    tf = T * dt
    t_hor = np.linspace(0, tf, T)

    if traj_show:
        
        for k in range(maxiters-1):
            u[:, -1, k] = u[:, -2, k]  # for plotting purpose

            # Create subplots for state trajectories
            fig, axs = plt.subplots(ns + 1, 1, figsize=(10, 3 * (ns + 1)))

            for i in range(ns):
                axs[i].plot(t_hor, x[i, :, k], linewidth=2, label=f'$x_{i+1}$')
                axs[i].plot(t_hor, xref[i, :], 'g--', linewidth=2, label=f'$x_{i+1}^{{ref}}$')
                axs[i].grid()
                axs[i].set_ylabel(f'$x_{i+1}$')
                axs[i].set_xlabel('Time')
                axs[i].set_title(f'State trajectory $x_{i+1}$ at iteration {k + 1}')
                axs[i].legend()

            # Create subplot for input trajectory
            axs[ns].plot(t_hor, u[0, :, k], 'r', linewidth=2, label='$u^*$')
            axs[ns].plot(t_hor, uref[0, :], 'r--', linewidth=2, label='$u^{ref}$')
            axs[ns].grid()
            axs[ns].set_ylabel('$u$')
            axs[ns].set_xlabel('Time')
            axs[ns].set_title(f'Input trajectory at iteration {k + 1}')
            axs[ns].legend()

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()
   
def traj_iteration_sameplot(x, u, xref, uref, maxiters, traj_show, dt, T):
    
    """
    This function is similar to the previous one, but allow to show trajectories at 
    different iterations in the same plot. In this function you can change if conditions
    to plot the trajectories at the desired iterations.
    """
    
    # Final time
    tf = T * dt
    t_hor = np.linspace(0, tf, T)

    if traj_show:
        # Create a single figure with subplots
        fig, axs = plt.subplots(ns + 1, 1, figsize=(10, 3 * (ns + 1)))

        # Plot all iterations on the same axes for state trajectories
        for i in range(ns):
            for k in range(maxiters):
                    # You can change the if condition to plot the desired iterations
                    if(k<=1 or k==9 or k==11):
                            axs[i].plot(t_hor, x[i, :, k], linewidth=1.5, label=f'Iteration {k + 1}')
                    elif(k==maxiters-1):
                            axs[i].plot(t_hor, x[i, :, k],color ='purple', linewidth=1.5, label='Optimal trajectory')
            axs[i].plot(t_hor, xref[i, :], '--',color='gray', linewidth=2, label=f'Reference curve $x_{i+1}$')
            axs[i].grid()
            axs[i].set_ylabel(f'$x_{i+1}$')
            axs[i].set_xlabel('Time')
            axs[i].set_title(f'State trajectory $x_{i+1}$ along iterations')
            axs[i].legend()

        # Plot all iterations on the same axis for the input trajectory
        for k in range(maxiters):
            u[:, -1, k] = u[:, -2, k]  # for plotting purpose
            # You can change the if condition to plot the desired iterations
            if(k<=1 or k==9 or k==11):
                axs[ns].plot(t_hor, u[0, :, k], linewidth=1.5,label=f'Iteration {k + 1}')
            elif(k==maxiters-1):
                axs[ns].plot(t_hor, u[0, :, k], color ='purple',linewidth=1.5, label='Optimal trajectory')
        axs[ns].plot(t_hor, uref[0, :], '--',color='gray', linewidth=2,label='Reference curve $u$')
        axs[ns].grid()
        axs[ns].set_ylabel('$u$')
        axs[ns].set_xlabel('Time')
        axs[ns].set_title('Input trajectory along iterations')
        axs[ns].legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()   


def Newton_reg(xref,uref,xinit,uinit,T,dt,maxiters,Q,R,QT,stepsize_0,beta,c,armijo_maxiters,show,traj_show,armijo_plot,term_cond = None,fixed_stepsize = None):

    """
    This function implements the regularized Newton's method to solve a trajectory generation problem
    for a discrete-time dynamical system. It iteratively computes the optimal state and input 
    trajectories that minimize a given cost function subject to system dynamics constraint.

    Inputs:
    - xref: 2D numpy array of reference state curves, shape (ns, T).
            Represents the desired state curve over the time horizon.
    - uref: 2D numpy array of reference input curve, shape (ni, T).
            Represents the desired input curve over the time horizon.
    - xinit: 2D numpy array, initial guess for the state trajectories, shape (ns, T).
    - uinit: 2D numpy array, initial guess for the input trajectory, shape (ni, T).
    - T: Integer, number of discrete time steps in the time horizon.
    - dt: Float, the discretization step.
    - maxiters: Integer, maximum number of iterations for the Newton's method.
    - Q: 3D numpy array or 2D array, state weighting matrix, shape (ns, ns, T) or (ns, ns).
    - R: 3D numpy array or 2D array, input weighting matrix, shape (ni, ni, T) or (ni, ni).
    - QT: 2D numpy array, terminal cost weighting matrix, shape (ns, ns).
    - stepsize_0: Float, initial step size for the Armijo line search.
    - beta: Float, step size reduction factor for the Armijo line search.
    - c: Float, parameter for the Armijo condition.
    - armijo_maxiters: Integer, maximum iterations for the Armijo line search.
    - show: Boolean, if True, plots the cost and descent direction norm after optimization.
    - traj_show: Boolean, if True, plots the state and input trajectories for each iteration.
    - armijo_plot: Boolean, if True, visualizes the Armijo discent direction plot.
    - term_cond: Float (optional), threshold for the norm of the descent direction to terminate iterations early.
    - fixed_stepsize: Float (optional), if provided, the step size is fixed to this value.

    Outputs:
    - x: 3D numpy array, optimal state trajectory at each iteration, shape (ns, T, maxiters).
    - u: 3D numpy array, optimal input trajectory at each iteration, shape (ni, T, maxiters).
    - maxiters: Integer, maximum number of iterations.
    """



    # x contains optimal state trajectory estimate at each iteration
    x = np.zeros((ns, T, maxiters))
    # u contains optimal input trajectory estimate at each iteration   
    u = np.zeros((ni, T, maxiters))
    # Definition of state linearization matrix A 
    AA = np.zeros((ns,ns,T))
    # Definition of input linearization matrix B
    BB = np.zeros((ns,ni,T))
    # Gradient of the cost wrt x
    q = np.zeros((ns,T))
    # Gradient of the cost wrt u
    r = np.zeros((ni,T))
    # lambda vector
    lmbd = np.zeros((ns, T, maxiters))
    # Cost J(u)
    J = np.zeros((maxiters))
    # Gradient of the cost function J wrt u
    grad_J = np.zeros((ni,T,maxiters))
    # Descent direction
    deltau = np.zeros((ni,T, maxiters))
    # Norm of the descent direction
    deltau_norm = np.zeros((maxiters))
    # If a matrix is not a 3D array (because constant, not function of time), add third dimension
    # and copy it for each time so that is constant at each time
    if Q.ndim < 3:
      Q = Q[:,:,None]
      Q = Q.repeat(T, axis=2)
    if R.ndim < 3:
      R = R[:,:,None]
      R = R.repeat(T, axis=2)

    # Inizialization of the Newton method
    # Initial guess for the optimal state/input trajectory
    x[:,:,0] = xinit
    u[:,:,0] = uinit

    # Initial condition
    x0 = xref[:,0]

    #Let's define the matrices useful for finding the descent direction
    # Considering the regularized version of the Newton method
    # According to our cost function, they are the same for each iteration k
    Q_tilde = Q
    R_tilde = R
    
    for k in range(maxiters-1):

        for t in range(0,T-1):
            
            # State linearization matrix at time t and iteration k
            AA[:,:,t] = dyn.A(x[:,t,k],u[:,t,k])
            # Input linearization matrix at time t and iteration k
            BB[:,:,t] = dyn.B(x[:,t,k],u[:,t,k])
            # Gradient of the cost wrt x
            q[:,t] = stagecost(x[:,t,k],u[:,t,k],xref[:,t],uref[:,t],Q[:,:,t],R[:,:,t])[1]
            # Gradient of the cost wrt u
            r[:,t] = stagecost(x[:,t,k],u[:,t,k],xref[:,t],uref[:,t],Q[:,:,t],R[:,:,t])[2]
        
        # Solve lqr affine problem at each iteration k
        qT = termcost(x[:,-1,k],xref[:,-1],QT)[1]
        K, sigma, deltau[:,:,k], *_ = lqr_affine(AA,BB,0,Q_tilde,R_tilde,q,r,QT,qT,T)

        # We need to solve the co-state/adjoint equation to find the gradient of J
        # We need the gradient of J for the armijo_stepsize function

        # Terminal condition for lambda
        lmbd[:,-1,k] = termcost(x[:,-1,k],xref[:,-1],QT)[1]
        
        # Solve the co-state equation backward in time and get the gradient of J wrt u
        for t in reversed(range(0,T-1)):
           
           lmbd[:,t,k] = AA[:,:,t].T@lmbd[:,t+1,k] + q[:,t]

           grad_J[:,t,k] = BB[:,:,t].T@lmbd[:,t+1,k] + r[:,t]
      
        ###########################################################################################
        # Computation of the cost J(u) at each iteration for armijo and plotting purpose
        J[k] = 0 # Initialized to zero
        
        # Sum of stage costs
        for t in range(0,T-1):
           J[k] = J[k] + stagecost(x[:,t,k],u[:,t,k],xref[:,t],uref[:,t],Q[:,:,t],R[:,:,t])[0]
        
        # Sum the terminal cost
        J[k] = J[k] + termcost(x[:,-1,k],xref[:,-1],QT)[0]
        ###########################################################################################

        if fixed_stepsize is None:
           # Stepsize chosen with armijo rule
           stepsize = armijo_stepsize(stepsize_0,beta,c,J[k],u[:,:,k],xref,uref,
                                      deltau[:,:,k],grad_J[:,:,k],armijo_maxiters,Q,R,QT,armijo_plot)
        
        else:
           # Stepsize chosen constant
           stepsize = fixed_stepsize
        

        # Compute the new state-input trajectory by forward integration (closed loop)
        x[:,0,k+1] = x0
        for t in range(0,T-1):
           u[:,t,k+1] = u[:,t,k] + K[:,:,t]@(x[:,t,k+1]-x[:,t,k]) +stepsize*sigma[:,t]
           x[:,t+1,k+1] = f(x[:,t,k+1],u[:,t,k+1])


           # Norm of the descent direction at each iteration k for plotting purpose
           deltau_norm[k] += deltau[:,t,k].T@deltau[:,t,k]
        
        # if the norm of the descent direction is sufficiently small, stop the iterations
        #print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(k,deltau_norm[k], J[k]))
        if term_cond is not None:
           if deltau_norm[k] < term_cond:
                  maxiters = k + 1
                  break
    
    
   # Plots of cost and descent direction norm
    if show:
      #Norm of the descent direction deltau 
      plt.figure('descent direction')
      plt.plot(np.arange(maxiters), deltau_norm[:maxiters])
      plt.xlabel('$k$')
      plt.ylabel('$\\|\\Delta \\mathbf{u}_k\\|$')
      plt.yscale('log')
      plt.title('Norm of the descent direction')
      plt.grid()
      plt.show(block=False)

      # Cost J(u) at each iteration
      plt.figure('cost')
      plt.plot(np.arange(maxiters), J[:maxiters], color = 'b')
      plt.xlabel('$k$')
      plt.ylabel('$J(\\mathbf{u}^k)$')
      plt.yscale('log')
      plt.title('Cost')
      plt.grid()
      plt.show(block=False)


    # Trajectories at each iteration of the Newton's method
    #This function plots the trajectories at each iteration in different plots
    traj_iteration_plot(x,u,xref,uref,maxiters,traj_show,dt,T)
    #This function plots the trajectories at each iteration in the same plot
    #traj_iteration_sameplot(x,u,xref,uref,maxiters,traj_show,dt,T)
    return x,u,maxiters



        