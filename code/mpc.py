import numpy as np
from dynamics import linearize_system,f,ns,ni
from LQR import lqr_mpc
import dynamics as dyn
import cvxpy as cp


def mpc_controller1(x_gen, u_gen, noise, T, Q, R, QT):
    """
    Simulates the system's behavior using Model Predictive Control (MPC). The function generates 
    a trajectory based on a reference trajectory (`x_gen`) and control input (`u_gen`) with a 
    horizon `T`. It performs control at each time step by solving an optimization problem for 
    a rolling horizon and applying control to the system while considering measurement noise.

    Args:
        Q, R: Weight matrices for state and input in the stage cost function
        QT: weight function for the terminal cost
        x_gen: np.array of shape (ns, time), the reference state trajectory [theta1, theta2, omega1, omega2].
        u_gen: np.array of shape (ni, time), the reference control input trajectory.
        noise: np.array of shape (ns,), representing measurement noise.
        T: Horizon length for MPC. It determines the number of steps used for prediction and optimization.

    Returns:
        x_real: np.array of shape (ns, time), real state trajectory after applying MPC control, with measurement noise.
        u_real: np.array of shape (ni, time), real input trajectory used for generating the state trajectory.
    """

    # Add measurement noise to the initial state
    x_meas = x_gen[:, 0] + noise  # Initial state with small random noise

    # Linearize the system around the reference trajectory (x_gen) and control inputs (u_gen)
    A, B = linearize_system(x_gen, u_gen)

    # Initialize lists for the real state trajectory and control input trajectory
    x_real = [x_meas]  # Starting with the noisy initial state
    u_real = []  # List to store control inputs applied at each time step

    # Simulate the system behavior with MPC for a rolling horizon
    for t in range(x_gen.shape[1] - T - 1):
        # Compute the LQR gain matrix for the current time step, based on the system's linearization
        
        Kp = lqr_mpc(A, B, t, T, Q, R, QT)

        # Calculate the control input
        u = u_gen[:, t] + Kp[0] @ (x_meas - x_gen[:, t])

        # Simulate the next state using the system's dynamics function `f`, and add measurement noise
        x_meas = f(x_meas, u)

        # Append the control input 
        u_real.append(u)

        #If we're not at the final time step, append the new measured state to the trajectory
        if t == x_gen.shape[1] - T - 1:
            break  # Stop at the final time step
        else:
            x_real.append(x_meas)  # Append the next measured state to the trajectory
        

    # Convert the trajectories to NumPy arrays with consistent dimensions
    x_real = np.array(x_real).T  # Shape: (ns, time)
    u_real = np.array(u_real).T  # Shape: (ni, time)

    # Ensure x_real and u_real have the same time dimension
    t_min = min(x_real.shape[1], u_real.shape[1])
    x_real = x_real[:, :t_min]
    u_real = u_real[:, :t_min]

    return x_real, u_real




def mpc_controller2(Q, R, QT, x_ref, u_ref, x0, T, pred_hor):
    """
    MPC controller for a nonlinear dynamic system using CVXPY.

    Parameters:
    - Q, R: Weight matrices for state and input in the cost function
    - x_ref, u_ref: State and input reference trajectory over the prediction horizon
    - x0: Initial state of the system
    - T: reference trajectory duration
    - pred_hor: prediction horizon

    Returns:
    - x_trajectory: Real state trajectory over the simulation steps 
    - u_trajectory: Real input trajectory over the simulation steps 
    """

    # Definition of state linearization matrix A 
    A = np.zeros((ns,ns,T))
    # Definition of input linearization matrix B
    B = np.zeros((ns,ni,T))

    for t in range(0,T-1):
            # State linearization matrix at time t 
            A[:,:,t] = dyn.A(x_ref[:,t],u_ref[:,t])
            # Input linearization matrix at time t
            B[:,:,t] = dyn.B(x_ref[:,t],u_ref[:,t])

    # Initialize storage for state and input trajectories
    x_trajectory = np.zeros((ns, T-pred_hor))
    u_trajectory = np.zeros((ni, T-pred_hor))

    x_trajectory[:, 0] = x0
    x_current = x0

    for t in range(T-pred_hor-1):
        # Define optimization variables
        x = cp.Variable((ns, pred_hor))  # States over the horizon
        u = cp.Variable((ni,pred_hor))     # Inputs over the horizon

        # Define the cost function
        cost = 0
        constraints = []

        for tau in range(t,t+pred_hor-1):
            # Quadratic cost for state and input deviations
            cost += cp.quad_form(x[:, tau-t] - x_ref[:, tau], Q)
            cost += cp.quad_form(u[:, tau-t] - u_ref[:, tau], R)

            # System dynamics constraint (linearized for prediction)
            constraints.append(x[:, tau-t + 1] == A[:,:,tau] @ x[:, tau-t] + B[:,:,tau] @ u[:, tau-t])

        # Terminal cost
        cost += cp.quad_form(x[:,-1]-x_ref[:,t+pred_hor-1], QT)
        # Initial state constraint
        constraints.append(x[:, 0] == x_current)

        # Formulate the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the problem
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("MPC optimization failed: " + problem.status)

        # Extract the optimal control input
        u_opt = u[:, 0].value
        u_trajectory[:, t] = u_opt

        # Update the current state based on the nonlinear dynamics
        x_next = f(x_current, u_opt)
        x_trajectory[:, t + 1] = x_next

        x_current = x_next

    return x_trajectory, u_trajectory