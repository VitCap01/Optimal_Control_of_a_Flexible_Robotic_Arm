import numpy as np
from cost import stagecost,termcost
from dynamics import ns,ni,f
import matplotlib.pyplot as plt

def armijo_stepsize(stepsize_0,beta,c,J_uk,u_k,xref,uref,deltau_k,gradJ_k,armijo_maxiters,Q,R,QT,descent_plot):

    """
    Perform Armijo line search to determine the optimal step size for a descent direction.

    Parameters:
    - stepsize_0: Initial step size (scalar).
    - beta: Reduction factor for the step size (0 < beta < 1).
    - c: Armijo condition constant (0 < c < 1).
    - J_uk: Cost at the current control input trajectory `u_k`.
    - u_k: Current control input trajectory (2D array, shape (ni, T)).
    - xref: Reference state curve (2D array, shape (ns, T)).
    - uref: Reference control input curve (2D array, shape (ni, T)).
    - deltau_k: Descent direction for the control input (2D array, shape (ni, T)).
    - gradJ_k: Gradient of the cost function with respect to the control input (2D array, shape (ni, T)).
    - armijo_maxiters: Maximum number of iterations for Armijo line search.
    - Q: State weighting matrix for stage cost (3D array, shape (ns, ns, T) or (ns, ns)).
    - R: Control weighting matrix for stage cost (3D array, shape (ni, ni, T) or (ni, ni)).
    - QT: Terminal cost weighting matrix (2D array, shape (ns, ns)).
    - descent_plot: Boolean flag to indicate whether to plot the Armijo descent direction plot.

    Returns:
    - stepsize: Optimal step size that satisfies the Armijo condition.
    """

    # Time horizon
    T = u_k.shape[1]
    # Inizialization of stepsize
    stepsize = stepsize_0
    # List of stepsizes
    stepsizes = []
    # List of costs J_gamma
    costs_arm = []

    # Scalar product netwwen gradient and descent direction
    m = 0
    for t in range(0,T-1):
        m = m + gradJ_k[:,t].T@deltau_k[:,t]

    # Adjust matrix dimansion if needed
    if Q.ndim < 3:
      Q = Q[:,:,None]
      Q = Q.repeat(T, axis=2)
    if R.ndim < 3:
      R = R[:,:,None]
      R = R.repeat(T, axis=2)

    for i in range(armijo_maxiters):

        # Cost computed at a certain stepsize gamma with fixed u_k and direction dir_k
        J_gamma = 0

        # temp solution update
        x_temp = np.zeros((ns,T))
        u_temp = np.zeros((ni,T))

        # Initial condition x0
        x_temp[:,0] = xref[:,0]


        for t in range(0,T-1):
            u_temp[:,t] = u_k[:,t] + stepsize*deltau_k[:,t]
            x_temp[:,t+1] = f(x_temp[:,t], u_temp[:,t])


        # Sum of stage costs
        for t in range(0,T-1):
        
            J_gamma = J_gamma + stagecost(x_temp[:,t],u_temp[:,t],xref[:,t],uref[:,t],Q[:,:,t],R[:,:,t])[0]
        
        # Sum the terminal cost
        J_gamma = J_gamma + termcost(x_temp[:,-1],xref[:,-1],QT)[0]

        # Collect stepsizes and costs
        stepsizes.append(stepsize)      
        #costs_arm.append(np.min([J_gamma, 100*J_uk]))
        costs_arm.append(J_gamma)
        
        
        if (J_gamma >= J_uk + c*stepsize*m):

            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            #print('Armijo stepsize = {:.3e}'.format(stepsize))
            break
            
        if i == armijo_maxiters - 1:
            print("WARNING: no stepsize was found with armijo rule!")

        
    ############################
    # Descent Plot
    ############################

    if descent_plot:

        steps = np.linspace(0,stepsize_0,int(3e1))
        costs = np.zeros(len(steps))

        for i in range(len(steps)):

                step = steps[i]

                # temp solution update
                xx_temp = np.zeros((ns,T))
                uu_temp = np.zeros((ni,T))

                #Initial condition
                xx_temp[:,0] = xref[:,0]

                for t in range(T-1):
                    uu_temp[:,t] = u_k[:,t] + step*deltau_k[:,t]
                    xx_temp[:,t+1] = f(xx_temp[:,t], uu_temp[:,t])

                # Cost computed at a certain stepsize gamma with fixed u_k and direction dir_k
                J_gamma = 0
                
                # Sum of stage costs
                for t in range(T-1):
                    J_gamma = J_gamma + stagecost(xx_temp[:,t], uu_temp[:,t], xref[:,t], uref[:,t],Q[:,:,t],R[:,:,t])[0]

                # Terminal cost
                J_gamma = J_gamma + termcost(xx_temp[:,-1], xref[:,-1],QT)[0]

                #costs[i] = np.min([J_gamma, 100*J_uk])
                costs[i] = J_gamma


        plt.figure(1)
        plt.clf()

        # Descent armijo plot
        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k + stepsize*d^k)$')
        plt.plot(steps, J_uk + m*steps, color='r', label='$J(\\mathbf{u}^k) + stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, J_uk + c*m*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) + stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

        # plot the tested stepsize
        plt.scatter(stepsizes, costs_arm, marker='*') 

        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()

        plt.show()

            
    return stepsize



    


    

