import numpy as np
import matplotlib.pyplot as plt
from dynamics import dt,a,b,c,d,e,f1,f2

def polynomial_transition(pos_start, pos_end, duration):
        """
        Generates a polynomial curve that starts exactly at `pos_start` and ends at `pos_end`
        within the specified `duration`. Ensures smooth and continuous transitions.

        Args:
            pos_start: Initial position.
            pos_end: Final position.
            duration: Number of discrete time steps.

        Returns:
            Array containing the polynomial curve over the specified `duration`.
        """
        # Create a normalized time array from 0 to 1 (independent of the duration)
        t_vals = np.linspace(0, 1, int(duration))

        # Calculate the trajectory using a cubic polynomial
        # Ensures that the initial and final velocities are zero
        trajectory = pos_start + (pos_end - pos_start) * (3 * t_vals**2 - 2 * t_vals**3)

        # Return the trajectory as a numpy array for compatibility
        return trajectory

def ref_curve_from_theta1(theta_1, t2, o2):


    """
    Generate a reference state-input curve for task 2 (swing-up) based given the reference 
    curve of theta_1 and initial conditions for theta_2 and related angular velocity.
    Based on that, it computes theta_2, omega_2 and u based on the Euler-Lagrange 
    dynamics equations.

    Inputs:
    - theta_1: reference curve for theta_1.
    - t2: initial condition for theta_2.
    - o2: initial condition for the angular velocity related to theta_1.

    Outputs:
    - x: state reference curve
    - u: input reference curve
    """

    def calculate_derivative(v, dt):
        """
        Calculates the derivative of a vector v with respect to time t with a constant time interval dt.
        For the edges, forward difference (for the first element)
        and backward difference (for the last element) are used.
        
        Inputs:
            v: data vector (size N).
            dt: constant time interval between elements of v.
            
        Outputs:
            derivative: vector of the same size as v containing the derivative.
        """
        
        # Create an array for the derivative of the same size as v
        derivative = np.zeros_like(v)
        
        # Calculate the derivative using central finite differences for the internal points
        for i in range(0, len(v) - 1):
            derivative[i] = (v[i + 1] - v[i]) / (dt)  # Central difference
        
        # Handle edges: finite differences at the beginning and end
        # Backward difference for the last point
        derivative[-1] = derivative[-2] 
        
        return derivative
    
    # Compute angular velocity and acceleration related to theta_1
    omega_1 = calculate_derivative(theta_1, dt)
    alpha_1 = calculate_derivative(omega_1, dt)

    # Initialize theta_2, omega_2, alpha_2, u
    theta_2 = np.zeros_like(theta_1)
    omega_2 = np.zeros_like(theta_1)
    alpha_2 = np.zeros_like(theta_1)
    u = np.zeros_like(theta_1)

    # Set initial conditions for theta_2 and omega_2
    omega_2[0] = o2
    theta_2[0] = t2

    # Loop for computing dynamics
    for t in range(len(theta_1)-1):
        # Update alpha_2, omega_2, and theta_2
        alpha_2[t] = -((c + b * np.cos(theta_2[t])) * alpha_1[t] + b * np.sin(theta_2[t]) * omega_1[t]**2 + f2 * omega_2[t] + e * np.sin(theta_1[t] + theta_2[t])) / c 
        omega_2[t+1] = omega_2[t] + dt * alpha_2[t]
        theta_2[t+1] = theta_2[t] + dt * omega_2[t] 

    # Compute control input u
    for t in range(len(theta_1)):
        u[t] = (a + 2*b * np.cos(theta_2[t])) * alpha_1[t] + (c + b * np.cos(theta_2[t])) * alpha_2[t] - b * omega_2[t] * np.sin(theta_2[t]) * (omega_2[t] + 2 * omega_1[t]) 
        u[t] += f1 * omega_1[t] + d * np.sin(theta_1[t]) + e * np.sin(theta_1[t] + theta_2[t])

    # Combine states into a list
    x = np.array([[theta_1[t], theta_2[t], omega_1[t], omega_2[t]] for t in range(len(theta_1))])


    return x,u

def reference_curve():

    """
    Generates a reference curve for task 2 (swing-up). First a reference curve for theta_1 is designed 
    considering one initial constant part at 0 rad, one final constant part at pi rad and a cubic 
    polynomial transition in between. According to this theta_1 reference curve, the function 
    ref_curve_from_theta1 is used to get the all state-input curve. Then this curve is elongated with a
    cubic polynomial transition to reach the upright equlibrium and a constant part.

    Inputs:
    None

    Returns:
    - xref (numpy.ndarray): State reference curve 
    - uref (numpy.ndarray): Input reference curve
    - T (int): Total number of discrete time steps in the reference curve.
    """

    # Discrete time samples for the parts of the reference curve for theta_1
    samples = 2000 
    added_portion = 500 
        
    # Define theta_1 curve
    theta_1 = np.full(samples+added_portion, 0)
    theta_1 = np.concatenate((theta_1, polynomial_transition(0, np.pi, 2066)))
    theta_1 = np.concatenate((theta_1, np.full(samples+added_portion, np.pi)))
        
        
    # Generate state-input reference curve based on theta_1
    xref, uref = ref_curve_from_theta1(theta_1, 0, 0)
        
        
    xref = xref[:len(xref)-added_portion]
    uref = uref[:len(uref)-added_portion]
    
    # Elongation of the reference curve
    # Polynomial transition to reach the upright equlibrium
    xref = np.concatenate([xref,np.array([
        polynomial_transition(xref[-1, 0], np.pi, added_portion),
        polynomial_transition(xref[-1, 1], 0, added_portion),
        polynomial_transition(xref[-1, 2], 0, added_portion),
        polynomial_transition(xref[-1, 3], 0, added_portion)
    ]).T], axis=0)
    uref = np.concatenate((uref, polynomial_transition(uref[-1], 0, added_portion)), axis=0)
    
    # Elongation of the reference curve with constant parts corresponding to the upright equilibrium
    for i in range(added_portion):
        xref = np.concatenate((xref, np.array([np.pi, 0, 0, 0]).reshape(1, -1)))
        uref = np.concatenate((uref, np.array([0])))

    # Number of discrete time samples
    T = len(xref)
    
    return xref,uref,T



    