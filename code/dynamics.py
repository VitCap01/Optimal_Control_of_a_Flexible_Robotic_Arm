import numpy as np

# discretization step
dt = 1e-3

# Number of states and inputs
ns = 4
ni = 1

# Set of parameters 3
m1 = 1.5
m2 = 1.5
l1 = 2
l2 = 2
r1 = 1
r2 = 1
I1 = 2
I2 = 2
g = 9.81
f1 = 0.1
f2 = 0.1

# Definition of some constants for more compact notation
a = I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2)
b = m2*l1*r2
c = I2 + m2*r2**2
d = g*(m1*r1 + m2*l1)
e = g*m2*r2

# Non linear dynamic model in state space
def f(x, u):

    """
    This function computes the nonlinear dynamic model in state-space form for the flexible robot
    describing the evolution of the state vector `x` based on the current state and control input `u`.

    Parameters:
    - x : State vector
    - u : Control input.

    Returns:
    - f : non linear dynamic model of the system.
    """

    f = np.zeros((ns,))
    
    x1 = x[0]  
    x2 = x[1]  
    x3 = x[2]  
    x4 = x[3]  
 
    f[0] = x1 + dt*x3
    f[1] = x2 + dt*x4
    f[2] = x3 - dt*c/(c*(a-c)-b**2*np.cos(x2)**2)*(b*x4*np.sin(x2)*(-x4-2*x3) + f1*x3 + d*np.sin(x1) + e*np.sin(x1+x2) - u)
    f[2] = f[2] + dt*(c + b*np.cos(x2))/(c*(a-c) - b**2*np.cos(x2)**2)*(b*np.sin(x2)*x3**2 + f2*x4 + e*np.sin(x1+x2))
    f[3] = x4 + dt*(c + b*np.cos(x2))/(c*(a-c) - b**2*np.cos(x2)**2)*(b*x4*np.sin(x2)*(-x4-2*x3) + f1*x3 + d*np.sin(x1) + e*np.sin(x1+x2) - u)
    f[3] = f[3] - dt*(a + 2*b*np.cos(x2))/(c*(a-c) - b**2*np.cos(x2)**2)*(b*np.sin(x2)*x3**2 + f2*x4 + e*np.sin(x1+x2))
 
    return f


# State linearization matrix
def A(x,u):
        
        """
        This function computes the Jacobian state linearization matrix (A) for the state-space system, which represents 
        the linearized dynamics of the system around the current state `x` and control input `u`. 
    
        The matrix `A` is calculated by taking the partial derivatives of the nonlinear system's 
        dynamic equations with respect to the states. 

        Parameters:
        - x : State vector
        - u : Control input .

        Returns:
        - A : State linearization matrix
        """

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
 
        def j_1_1(x1,x2,x3,x4):
            return 1
 
        def j_1_2(x1,x2,x3,x4):
            return 0
 
        def j_1_3(x1,x2,x3,x4):
            return dt
 
        def j_1_4(x1,x2,x3,x4):
            return 0
 
 
        def j_2_1(x1,x2,x3,x4):
            return 0
 
        def j_2_2(x1,x2,x3,x4):
            return 1
 
        def j_2_3(x1,x2,x3,x4):
            return 0
 
        def j_2_4(x1,x2,x3,x4):
            return dt
 
 
        def j_3_1(x1,x2,x3,x4):
            res = - dt*c/(c*(a-c)-b**2*np.cos(x2)**2)*(d*np.cos(x1)+ e*np.cos(x1+x2))
            res += dt*(c + b*np.cos(x2))/(c*(a-c) - b**2*np.cos(x2)**2)*(e*np.cos(x1+x2))
            return res
 
        def j_3_2(x1,x2,x3,x4,u):
            res = dt*(2*b**2*c*np.sin(x2)*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)**2*(b*x4*np.sin(x2)*(-x4-2*x3)+ f1*x3+ d*np.sin(x1)+ e*np.sin(x1+x2)-u)
            res = res - dt*c/(c*(a-c)-b**2*np.cos(x2)**2 )*(b*x4*np.cos(x2)*(-x4-2*x3)+e*np.cos(x1+x2) )
            res = res - dt*(b*np.sin(x2)*(c*(a-c)+b**2* np.cos(x2)**2+2*b*c*np.cos(x2)))/(c*(a-c)-b**2*np.cos(x2)**2)**2 *(b*np.sin(x2)*x3**2+f2*x4+ e*np.sin(x1+x2))
            res = res + dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(b*np.cos(x2)*x3**2+e*np.cos(x1+x2))
            return res
 
        def j_3_3(x1,x2,x3,x4):
            res = 1 - dt*c/(c*(a-c)-b**2*np.cos(x2)**2)*(-2*b*x4*np.sin(x2)+f1)
            res = res + dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(2*b*np.sin(x2)*x3)
            return res
 
        def j_3_4(x1,x2,x3,x4):
            res = - dt*c/(c*(a-c)-b**2*np.cos(x2)**2)*(-2*b*x4*np.sin(x2)-2*b*x3*np.sin(x2))
            res = res + dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(f2)
            return res
 
 
        def j_4_1(x1,x2,x3,x4):
            res = dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(d*np.cos(x1)+ e*np.cos(x1+x2))
            res += - dt*(a + 2*b*np.cos(x2))/(c*(a-c) - b**2*np.cos(x2)**2)*(e*np.cos(x1+x2))
            return res
 
        def j_4_2(x1,x2,x3,x4,u):
            res = dt*(((-b*np.sin(x2))*(c*(a-c)-b**2*np.cos(x2)**2)-(2*b**2*np.sin(x2)*np.cos(x2))*(c+b*np.cos(x2)))/(c*(a-c)-b**2*np.cos(x2)**2)**2)*(b*x4*np.sin(x2)*(-x4-2*x3)+f1*x3+ d*np.sin(x1)+e*np.sin(x1+x2)-u)  
            res = res + dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(b*x4*np.cos(x2)*(-x4-2*x3)+e*np.cos(x1+x2)) #ok
            res = res + dt*(2*b*np.sin(x2)*(b*np.cos(x2)+c)*(a+b*np.cos(x2)-c))/(c*(a-c)-b**2*np.cos(x2)**2)**2*(b*np.sin(x2)*x3**2+f2*x4+e*np.sin(x1+x2) )
            res = res - dt*(a+2*b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(b*np.cos(x2)*x3**2+e*np.cos(x1+x2)) #ok
            return res
 
        def j_4_3(x1,x2,x3,x4):
            res = dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(-2*b*x4*np.sin(x2)+ f1)
            res = res - dt*(a+2*b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(2*b*np.sin(x2)*x3)
            return res
 
        def j_4_4(x1,x2,x3,x4):
            res = 1 + dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(-2*b*x4*np.sin(x2)- 2*b*x3*np.sin(x2))
            res = res - dt*(a+2*b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)*(f2)
            return res
 
        A = np.zeros((ns,ns))
        A[0,0] = j_1_1(x1,x2,x3,x4)
        A[0,1] = j_1_2(x1,x2,x3,x4)
        A[0,2] = j_1_3(x1,x2,x3,x4)
        A[0,3] = j_1_4(x1,x2,x3,x4)


        A[1,0] = j_2_1(x1,x2,x3,x4)
        A[1,1] = j_2_2(x1,x2,x3,x4)
        A[1,2] = j_2_3(x1,x2,x3,x4)
        A[1,3] = j_2_4(x1,x2,x3,x4)

        A[2,0] = j_3_1(x1,x2,x3,x4)
        A[2,1] = j_3_2(x1,x2,x3,x4,u)
        A[2,2] = j_3_3(x1,x2,x3,x4)
        A[2,3] = j_3_4(x1,x2,x3,x4)

        A[3,0] = j_4_1(x1,x2,x3,x4)
        A[3,1] = j_4_2(x1,x2,x3,x4,u)
        A[3,2] = j_4_3(x1,x2,x3,x4)
        A[3,3] = j_4_4(x1,x2,x3,x4)
       
        return A


# Input linearization matrix
def B(x,u):

        """
        This function computes the Jacobian input linearization matrix (B) for the state-space system, which represents 
        the linearized dynamics of the system around the current state `x` and control input `u`. 
    
        The matrix `B` is calculated by taking the partial derivatives of the nonlinear system's 
        dynamic equations with respect to the input. 

        Parameters:
        - x : State vector
        - u : Control input .

        Returns:
        - B : Input linearization matrix
        """

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
 
        b1 = 0
        b2 = 0
        b3 = dt*c/(c*(a-c)-b**2*np.cos(x2)**2)
        b4 = -dt*(c+b*np.cos(x2))/(c*(a-c)-b**2*np.cos(x2)**2)
 
        return np.array([[b1],[b2],[b3],[b4]])



def linearize_system(x_ref,u_ref):
    
    """
    This function computes the linearized system matrices (A and B) around a reference 
    trajectory of states (`x_ref`) and control inputs (`u_ref`). The matrices are computed 
    for each time step in the trajectory and represent the Jacobian matrices of the state-space 
    system.

    Parameters:
    - x_ref : numpy.ndarray
        A 2D array of shape (ns, T), where `ns` is the number of states and `T` is the number of 
        time steps. Each column corresponds to the state vector at a specific time step.
    - u_ref : numpy.ndarray
        A 2D array of shape (ni, T), where `ni` is the number of inputs and `T` is the number of 
        time steps. Each column corresponds to the control input vector at a specific time step.

    Returns:
    - Amatrix : numpy.ndarray
        State linearization matrix around a reference trajectory (T, ns, ns)
    - Bmatrix : numpy.ndarray
        Input linearization matrix around a reference trajectory of shape (T, ns, ni)
    """

    Amatrix = np.zeros((x_ref.shape[1],ns, ns))
    Bmatrix = np.zeros((u_ref.shape[1],ns, ni))

    for t in range(x_ref.shape[1]):
        Amatrix[t] = A(x_ref[:,t],u_ref[:,t]) 
        Bmatrix[t] = B(x_ref[:,t],u_ref[:,t]) 

    return Amatrix,Bmatrix



def linearize_system_position(x_pos,u_pos):

    """
    This function computes the linearized system matrices (A and B) around a reference 
    equilibrium position of states (`x_pos`) and control inputs (`u_pos`). 

    Parameters:
    - x_pos : numpy.ndarray
        Stete equlibrium
    - u_pos : numpy.ndarray
        input equilibrium

    Returns:
    - Amatrix : numpy.ndarray
        State linearization matrix around an equilibrium 
    - Bmatrix : numpy.ndarray
        Input linearization matrix around an equilibrium
    """

    Amatrix = np.zeros((ns, ns))
    Bmatrix = np.zeros((ns, ni))

    Amatrix = A(x_pos,u_pos)
    Bmatrix = B(x_pos,u_pos)

    return Amatrix,Bmatrix