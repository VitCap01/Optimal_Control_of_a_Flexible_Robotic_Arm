import numpy as np
import dynamics as dyn
from ref_curve_task2 import polynomial_transition

# Reference curve
def ref(step_reference,T,x1_eq1_deg,x1_eq2_deg):

      """
      Description:
      This function generates a state-input reference curve, transitioning between two equilibrium points. 
      The reference can be configured to follow either a smooth cubic polynomial transition or a step one.

      Inputs:
      - step_reference (bool): Determines the type of reference trajectory. 
      - If True, generates a step reference.
      - If False, generates a smooth transition using cubic polynomials.
      - T (int): Total time steps for the reference trajectory.
      - x1_eq1_deg (float): Initial equilibrium angle in degrees for state variable x1.
      - x1_eq2_deg (float): Final equilibrium angle in degrees for state variable x1.

      Outputs:
      - xref (numpy.ndarray): 
        array of shape (ns, T), containing the state reference curves for all states over time.
      
      - uref (numpy.ndarray): 
        array of shape (ni, T), containing the input reference curve over time.
      
      """

      # Number of states
      ns = dyn.ns
      # Number of inputs
      ni = dyn.ni
      # Coefficient d from dynamics for input equlibrium
      d = dyn.d
      
      # Inizialization of state/input references
      # velocities x3 and x4 will reamain zero since 
      # we're considering a trajectory between 2 equilibria
      xref = np.zeros((ns, T))
      uref = np.zeros((ni, T))

      # Let's convert in radians
      x1_eq1 = np.deg2rad(x1_eq1_deg)
      x1_eq2 = np.deg2rad(x1_eq2_deg)
      
      # Reference curve with smooth transition
      if not step_reference:
            
            # Define the duration for each segment
            t_transition = int(T / 4)  # Shorter smooth transition
            t_const = (T - t_transition) // 2  # Equal constant parts before and after transition

            # First constant part
            
            # x1ref
            xref[0, :t_const] = x1_eq1
            # x2ref
            xref[1, :t_const] = -x1_eq1
            # uref
            uref[0, :t_const] = d * np.sin(x1_eq1)

            # Smooth cubic polynomial transition
                  
            # Smooth transition for x1ref
            xref[0, t_const:t_const+t_transition] = polynomial_transition(x1_eq1, x1_eq2, t_transition)
                  
            # Smooth transition for x2ref
            xref[1, t_const:t_const+t_transition] = polynomial_transition(-x1_eq1, -x1_eq2, t_transition)
                  
            # Smooth transition for uref
            uref[0, t_const:t_const+t_transition] = polynomial_transition(d * np.sin(x1_eq1), d * np.sin(x1_eq2),t_transition)

            # Final constant part
            
            # x1ref
            xref[0, t_const + t_transition:] = x1_eq2
            # x2ref
            xref[1, t_const + t_transition:] = -x1_eq2
            # uref     
            uref[0, t_const + t_transition:] = d * np.sin(x1_eq2)
      
      # Step reference
      else:
                        
            # first constant part of step reference
            
            # x1ref
            xref[0,:int(T/2)] = x1_eq1
            # x2ref
            xref[1,:int(T/2)] = -x1_eq1
            # uref
            uref[0,:int(T/2)] = d*np.sin(x1_eq1)
            
            # second constant part of step reference
            
            # x1ref
            xref[0,int(T/2):] = x1_eq2
            # x2ref
            xref[1,int(T/2):] = -x1_eq2
            # uref
            uref[0,int(T/2):] = d*np.sin(x1_eq2)


      return xref, uref