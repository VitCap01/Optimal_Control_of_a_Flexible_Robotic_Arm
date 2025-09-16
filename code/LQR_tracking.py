import numpy as np
import dynamics as dyn
from dynamics import f
from LQR import lqr

def lqr_track(xtraj,utraj,T,Qreg,Rreg,QTreg,x0):
    
    """
    This function implements a Linear Quadratic Regulator (LQR) tracking controller for a nonlinear system.
    It computes the optimal state and input trajectories to follow a given reference trajectory 
    using feedback control based on the linearization of the system dynamics.

    Inputs:
    - xtraj: 2D numpy array, reference state trajectory, shape (ns, T).
            Represents the desired state trajectory to track.
    - utraj: 2D numpy array, reference input trajectory, shape (ni, T).
            Represents the desired input trajectory to track.
    - T: Integer, number of discrete time steps in the time horizon.
    - Qreg: 3D numpy array or 2D array, state regulation weighting matrix, shape (ns, ns, T) or (ns, ns).
            Represents the penalty on state deviation at each time step.
    - Rreg: 3D numpy array or 2D array, input regulation weighting matrix, shape (ni, ni, T) or (ni, ni).
            Represents the penalty on input deviation at each time step.
    - QTreg: 2D numpy array, terminal state regulation weighting matrix, shape (ns, ns).
            Represents the penalty on the final state deviation.
    - x0: 1D numpy array, initial state vector, shape (ns,).
        Represents the initial condition of the system (even different with respect to the one of the reference).

    Outputs:
    - x: 2D numpy array, real state trajectory of the system, shape (ns, T).
        Represents the state trajectory resulting from applying the LQR tracking controller.
    - u: 2D numpy array, real input trajectory of the system, shape (ni, T).
        Represents the feedback control inputs used to follow the reference trajectory.
    """

    # Number of states
    ns = dyn.ns
    # Number of inputs
    ni = dyn.ni
    # State linearization matrix around the trajectory (xtraj,utraj)
    A = np.zeros((ns,ns,T))
    # Input linearization matrix around the trajectory (xtraj,utraj)
    B = np.zeros((ns,ni,T))
    # Control input
    u = np.zeros((ni,T))
    # State of the system
    x = np.zeros((ns,T))
    
    # Linearization matrices
    for t in range(T-1):
        
        A[:,:,t] = dyn.A(xtraj[:,t],utraj[:,t])
        B[:,:,t] = dyn.B(xtraj[:,t],utraj[:,t])
    
    # Optimal feedback gain
    Kreg,_= lqr(A,B,Qreg,Rreg,QTreg,T)

    # Initial condition x0 for the system
    x[:,0] = x0

    for t in range(T-1):

        # Apply the feedback controller to the (non linear) system 
        u[:,t] = utraj[:,t] + Kreg[:,:,t]@(x[:,t]-xtraj[:,t])
        # Run the (non linear) system
        x[:,t+1] = f(x[:,t],u[:,t])

    return x,u



    

