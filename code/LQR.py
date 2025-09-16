import numpy as np
import dynamics as dyn

# Number of states
ns = dyn.ns
# Number of inputs
ni = dyn.ni

# Linear time variant quadratic regulator (only quadratic terms, not affine)
# (the matrix S is optional)
def lqr(A,B,Q,R,QT,T,S = None):

    """
    This function solves the Linear Quadratic (LQ) problem with only quadratic terms (not affine) for a discrete-time system 
    over a finite time horizon. It computes the optimal feedback gain matrix K and the matrix P 
    by solving the difference Riccati equation backward in time.

    Inputs:
    - A: 3D numpy array or 2D array, system dynamics matrix, shape (ns, ns, T) or (ns, ns).
        Represents the state transition matrix at each time step.
    - B: 3D numpy array or 2D array, input matrix, shape (ns, ni, T) or (ns, ni).
        Represents the control input matrix at each time step.
    - Q: 3D numpy array or 2D array, state weighting matrix, shape (ns, ns, T) or (ns, ns).
         Represents the state penalty matrix at each time step.
    - R: 3D numpy array or 2D array, input weighting matrix, shape (ni, ni, T) or (ni, ni).
         Represents the input penalty matrix at each time step.
    - QT: 2D numpy array, terminal cost weighting matrix, shape (ns, ns).
          Represents the penalty on the final state.
    - T: Integer, number discrete time steps in the time horizon.
    - S: 3D numpy array or 2D array (optional), cross-term weighting matrix, shape (ni, ns, T) or (ni, ns).
        If provided, includes cross terms in the cost function.

    Outputs:
    - K: 3D numpy array, optimal gain matrix, shape (ni, ns, T).
        Represents the optimal feedback gain matrix at each time step.
    - P: 3D numpy array, shape (ns, ns, T).
        Represents the solution of the Riccati equation at each time step.
    """


    # P matrix
    P = np.zeros((ns,ns,T))
    # K gain matrix
    K = np.zeros((ni,ns,T))

    # If a matrix is not a 3D array (because constant, not function of time), add third dimension
    # and copy it for each time so that is constant at each time
    if A.ndim < 3:
      A = A[:,:,None]
      A = A.repeat(T, axis=2)
    if B.ndim < 3:
      B = B[:,:,None]
      B = B.repeat(T, axis=2)
    if Q.ndim < 3:
      Q = Q[:,:,None]
      Q = Q.repeat(T, axis=2)
    if R.ndim < 3:
      R = R[:,:,None]
      R = R.repeat(T, axis=2)
    if S is not None:
       if S.ndim < 3:
            S = S[:,:,None]
            S = S.repeat(T, axis=2)

    # Terminal condition PT = QT
    P[:,:,-1] = QT
    # Solve Difference Riccati Equation
    for t in reversed(range(0, T-1)):
        
        # Definition of matrices for compactness
        Pt = P[:,:,t]
        Ptplus = P[:,:,t+1]
        Qt = Q[:,:,t]
        Rt = R[:,:,t]
        At = A[:,:,t]
        Bt = B[:,:,t]
        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)

        # Difference Riccati Equation
        if S is None:
            P[:,:,t] = At.T@Ptplus@At -At.T@Ptplus@Bt@M_inv@Bt.T@Ptplus@At + Qt
        else:
            St = S[:,:,t]
            P[:,:,t] = At.T @ Ptplus @ At - (Bt.T@Ptplus@At + St).T @ M_inv @ (Bt.T@Ptplus@At + St) + Qt

    # Evaluate optimal gain matrix K
    for t in range(0,T-1):
        Ptplus = P[:,:,t+1]
        Rt = R[:,:,t]
        At = A[:,:,t]
        Bt = B[:,:,t]
        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)

        # Optimal gain matrix K
        if S is None:
            K[:, :, t] = -M_inv@Bt.T@Ptplus@At
        else:
            St = S[:,:,t]
            K[:, :, t] = -M_inv@(St + Bt.T@Ptplus@At)
 
    return K,P


# Linear time variant quadratic regulator (also affine terms in the cost, not in the dynamics)
def lqr_affine(A,B,x0,Q,R,q,r,QT,qT,T,S = None):
    
    """
    This function solves a Linear Quadratic (LQ) problem with affine terms in the cost function.
    It computes the optimal feedback gain matrix, the term sigma, and the optimal input trajectory
    and the related state trajectory by solving backward in time the difference Riccati equation.

    Inputs:
    - A: 3D numpy array or 2D array, system dynamics matrix, shape (ns, ns, T) or (ns, ns).
        Represents the state transition matrix at each time step.
    - B: 3D numpy array or 2D array, input matrix, shape (ns, ni, T) or (ns, ni).
        Represents the control input matrix at each time step.
    - x0: 1D numpy array, initial state vector, shape (ns,).
    - Q: 3D numpy array or 2D array, state weighting matrix, shape (ns, ns, T) or (ns, ns).
        Represents the state penalty matrix at each time step.
    - R: 3D numpy array or 2D array, input weighting matrix, shape (ni, ni, T) or (ni, ni).
        Represents the input penalty matrix at each time step.
    - q: 2D numpy array, state-related affine cost term, shape (ns, T).
    - r: 2D numpy array, input-related affine cost term, shape (ni, T).
    - QT: 2D numpy array, terminal cost weighting matrix, shape (ns, ns).
          Represents the penalty on the final state.
    - qT: 1D numpy array, terminal affine cost term, shape (ns,).
    - T: Integer, number of discrete time steps in the time horizon.
    - S: 3D numpy array or 2D array (optional), cross-term weighting matrix, shape (ni, ns, T) or (ni, ns).
        If provided, includes cross terms in the cost function.

    Outputs:
    - K: 3D numpy array, optimal feedback gain matrices, shape (ni, ns, T).
        Represents the feedback gain matrix at each time step.
    - sigma: 2D numpy array, feedforward terms, shape (ni, T).
    - u: 2D numpy array, optimal input trajectory, shape (ni, T).
    - x: 2D numpy array, optimal state trajectory, shape (ns, T).
    - P: 3D numpy array, shape (ns, ns, T).
        Represents the solution of the Riccati equation at each time step.
    """

    # P matrix
    P = np.zeros((ns,ns,T))
    # K gain matrix
    K = np.zeros((ni,ns,T))
    # Sigma
    sigma = np.zeros((ni,T))
    # p
    p = np.zeros((ns,T))
    # State trajectory
    x = np.zeros((ns, T))
    # Input trajectory
    u = np.zeros((ni, T))
    # Initial condition
    x[:,0] = x0

    # If not a 3D array (because constant, not function of time), add third dimension
    # and copy it for each time so that is constant at each time
    if A.ndim < 3:
      A = A[:,:,None]
      A = A.repeat(T, axis=2)
    if B.ndim < 3:
      B = B[:,:,None]
      B = B.repeat(T, axis=2)
    if Q.ndim < 3:
      Q = Q[:,:,None]
      Q = Q.repeat(T, axis=2)
    if R.ndim < 3:
      R = R[:,:,None]
      R = R.repeat(T, axis=2)
    if S is not None:
       if S.ndim < 3:
            S = S[:,:,None]
            S = S.repeat(T, axis=2)
    # If not a 2D array (because constant, not function of time), add second dimension
    # and copy it for each time so that is constant at each time
    if q.ndim < 2:
            q = r[:,None]
            q = q.repeat(T, axis=1)
    if r.ndim < 2:
            r = r[:,None]
            r = r.repeat(T, axis=1)

    # Terminal condition PT = QT
    P[:,:,-1] = QT
    # Terminal condition PT = QT
    p[:,-1] = qT

    # Solve Difference Riccati Equation bacward in time
    for t in reversed(range(0, T-1)):
        
        # Definition of matrices for compactness
        Pt = P[:,:,t]
        Ptplus = P[:,:,t+1]
        pt = p[:,t]
        ptplus = p[:,t+1]
        Qt = Q[:,:,t]
        Rt = R[:,:,t]
        qt = q[:,t]
        rt = r[:,t]
        At = A[:,:,t]
        Bt = B[:,:,t]
        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)
        mt = rt + Bt.T @ ptplus

    
        # Difference Riccati Equation
        if S is None:
            P[:,:,t] = At.T@Ptplus@At -At.T@Ptplus@Bt@M_inv@Bt.T@Ptplus@At + Qt
        else:
            St = S[:,:,t]
            P[:,:,t] = At.T @ Ptplus @ At - (Bt.T@Ptplus@At + St).T @ M_inv @ (Bt.T@Ptplus@At + St) + Qt
        
        # Solve the other equation backward in time
        if S is None:
            p[:,t] = At.T @ ptplus - (Bt.T@Ptplus@At).T @ M_inv @ mt + qt
        else:
            p[:,t] = At.T @ ptplus - (Bt.T@Ptplus@At).T @ M_inv @ mt + qt
        
    # Evaluate optimal gain matrix K and sigma
    for t in range(0,T-1):
        Ptplus = P[:,:,t+1]
        ptplus = p[:,t+1]
        Rt = R[:,:,t]
        At = A[:,:,t]
        Bt = B[:,:,t]
        qt = q[:,t]
        rt = r[:,t]
        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)
        mt = rt + Bt.T @ ptplus

        # Optimal gain matrix K
        if S is None:
            K[:, :, t] = -M_inv@Bt.T@Ptplus@At
        else:
            St = S[:,:,t]
            K[:, :, t] = -M_inv@(St + Bt.T@Ptplus@At)
        
        # Sigma
        sigma[:,t] = -M_inv@mt
    
    for t in range(T - 1):
      # Trajectory
      u[:, t] = K[:,:,t]@x[:, t] + sigma[:,t]
      x[:,t+1] = A[:,:,t]@x[:,t] + B[:,:,t]@u[:, t]

    return K,sigma,u,x,P 


def lqr_mpc(A,B,t,T,Q,R,QT,S = None):

    """
    This function is a variation of lqr function used in mpc_controller1 function
    to compute the right linearization matrices in the considered prediction horizon
    """

    # Number of states
    ns = 4
    # Number of inputs
    ni = 1
    # P matrix
    P = np.zeros((T,ns,ns))
    # K gain matrix
    K = np.zeros((T,ns))

    remaining_horizon = min(T, len(A) - t)
    A_matrix = A[t:t+remaining_horizon]
    B_matrix = B[t:t+remaining_horizon]
    
    # Terminal condition PT = QT
    P[-1] = QT

    # Solve Difference Riccati Equation
    for x in reversed(range(0, T-1)):
        
        # Definition of matrices for compactness
        Ptplus = P[x+1]
        Qt = Q
        Rt = R
        At = A_matrix[x]
        Bt = B_matrix[x]

        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)

        P[x] = At.T@Ptplus@At -At.T@Ptplus@Bt@M_inv@Bt.T@Ptplus@At + Qt

    # Evaluate optimal gain matrix K
    for x in range(0,T-1):

        # Definition of matrices for compactness
        Ptplus = P[x+1]
        Rt = R
        At = A_matrix[x]
        Bt = B_matrix[x]
        M_inv = np.linalg.inv(Rt + Bt.T@Ptplus@Bt)

        # Optimal gain matrix K
        K[x] = -M_inv@Bt.T@Ptplus@At

    return K