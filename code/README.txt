# README.txt


# Project Description:

This project focuses on designing an optimal trajectory for a flexible robotic arm, 
exploiting Newton's method for optimization and LQR/MPC controllers to track this trajectory.

# File Organization:

The project is organized into multiple Python files, each handling specific aspects of the project:

- **animation.py**: 
  Contains function to create an animation of the robot.

- **armijo.py**: 
  Contains function that implements the Armijo line search algorithm used for step size selection in the Newton's method.

- **cost.py**: 
  Contains functions that implement the stage cost and terminal cost.

- **dynamics.py**: 
  Includes functions to define and compute the non linear discrete tiem system's dynamics, and the linearization

- **LQR_tracking.py**: 
  Implements Linear Quadratic Regulator (LQR) for tracking a reference trajectory.

- **LQR.py**: 
  Implements the solution of the LQ problem.

- **mpc.py**: 
  Contains two different implementations of the basic Model Predictive Control (MPC) scheme

- **plots.py**: 
  Contains functions for plotting simulation results.

- **ref_curve_task1.py**: 
  Generates the reference curve specific to Task 1.

- **ref_curve_task2.py**: 
  Generates the reference curve specific to Task 5.

- **Reg_Newton_Method.py**: 
  Implements the regularized Newton's method.

The main where required tasks are executed are realized with python notebooks: one for task 1 and another
for the remaining tasks from 2 to 5

- **main_task_1.ipynb**: 
  Jupyter notebook for running and analyzing the simulation for Task 1, including parameter tuning and visualization.

- **main_task_2-5.ipynb**: 
  Jupyter notebook for running and analyzing simulations for Tasks 2 through 5. Includes detailed results and performance analysis.




# Usage Instructions:

The following boolean variable can be used:

In the Newton_reg function:

- show: if True, it shows the cost and descent direction norm along iterations 

- traj_show: if True, it shows trajectories at each iteration of the Newton method

- descent_plot: if True, it shows the Armijo descent direction plot

In the ref function:

- step_ref: if True, returns a step reference curve, otherwise a cubic polynomial transition between 2 constant parts
