import numpy as np
import dynamics as dyn

# Import number of states and inputs from dynamics
ns = dyn.ns
ni = dyn.ni

# Stage cost
def stagecost(x,u,x_ref,u_ref,Qt,Rt):
  
  """
    This function computes the stage cost, gradients, and Hessians of the cost for
    a given state and input according to a certain reference state-input trajectory
    and weighting matrices 

    Parameters:
    - x: Current state.
    - u: Current control input.
    - x_ref: Reference state curve
    - u_ref: Reference input curve
    - Qt: State weighting matrix.
    - Rt: Control input ing matrix.

    Returns:
    - lt: Stage cost (scalar).
    - dlx: Gradient of the cost with respect to the state `x`.
    - dlu: Gradient of the cost with respect to the control `u`.
    - ddlx: Hessian of the cost with respect to the state `x` .
    - ddlu: Hessian of the cost with respect to the control `u`.
    - ddlxu: Cross-term Hessian of the cost.
  """

  # Stage cost
  lt = 0.5*(x - x_ref).T@Qt@(x - x_ref) + 0.5*(u - u_ref).T@Rt@(u - u_ref)

  # Gradient wrt x
  dlx = Qt@(x - x_ref)
  # Gradient wrt u
  dlu = Rt@(u - u_ref)
  # Hessian wrt x
  ddlx = Qt
  # Hessian wrt u
  ddlu = Rt
  # Hessian wrt x and u
  ddlxu = 0

  return lt.squeeze(), dlx, dlu, ddlx, ddlu, ddlxu

# Terminal cost
def termcost(xT,xT_ref,QT):
  
  """
    This function computes the terminal cost, its gradient, and its Hessian for a given terminal state.
    
    Parameters:
    - xT     : Final state vector at the terminal time step.
    - xT_ref : Reference state vector at the terminal time step.
    - QT     : Terminal state weight matrix.
    
    Returns:
    - lT     : Terminal cost (scalar).
    - dlTxT  : Gradient of the terminal cost with respect to the final state `xT'.
    - ddlTxT : Hessian of the terminal cost with respect to the final state `xT`.
  """

  # Terminal cost
  lT = 0.5*(xT - xT_ref).T@QT@(xT - xT_ref)

  # Gradient (wrt xT)
  dlTxT = QT@(xT - xT_ref)
  # Hessian (wrt xT)
  ddlTxT = QT

  return lT.squeeze(), dlTxT, ddlTxT