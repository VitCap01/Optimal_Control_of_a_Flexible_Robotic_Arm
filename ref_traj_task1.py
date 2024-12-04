import numpy as np
import dynamics as dyn

# Function to generate a smooth cosine transition
def smooth_cosine_transition(t, t_start, t_end, start_pos, end_pos):
    
    alpha = (t - t_start) / (t_end - t_start)  # Normalized time in [0, 1]
    return (1 - 0.5 * (1 - np.cos(np.pi * alpha))) * start_pos + (0.5 * (1 - np.cos(np.pi * alpha))) * end_pos

# Reference trajectory
def ref(step_reference,T,x1_eq1_deg,x1_eq2_deg):

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

            # Smooth transition
            for t in range(t_const, t_const + t_transition):
                  
                  # Smooth transition for x1ref
                  xref[0, t] = smooth_cosine_transition(t, t_const, t_const + t_transition, x1_eq1, x1_eq2)
                  
                  # Smooth transition for x2ref
                  xref[1, t] = smooth_cosine_transition(t, t_const, t_const + t_transition, -x1_eq1, -x1_eq2)
                  
                  # Smooth transition for uref
                  uref[0, t] = smooth_cosine_transition(t, t_const, t_const + t_transition, d * np.sin(x1_eq1), d * np.sin(x1_eq2))

            # Final constant part
            
            # x1ref
            xref[0, t_const + t_transition:] = x1_eq2
            # x2ref
            xref[1, t_const + t_transition:] = -x1_eq2
            # uref     
            uref[0, t_const + t_transition:] = d * np.sin(x1_eq2)

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