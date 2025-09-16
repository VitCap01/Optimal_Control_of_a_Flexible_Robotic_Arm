import matplotlib.pyplot as plt

# Trajectory generation plots
def traj_gen_plot(tt_hor, xstar, xref, ustar, uref, ns):
    """
    Plots the optimal state and input trajectories vs reference curves.

    Parameters:
    tt_hor : numpy array
        Time horizon array.
    xstar : numpy array
        Optimal state trajectory (shape: ns x T).
    xref : numpy array
        Reference state curve (shape: ns x T).
    ustar : numpy array
        Optimal input trajectory (shape: 1 x T).
    uref : numpy array
        Reference input curve (shape: 1 x T).
    ns : int
        Number of states.
    """
    # Set up the color palette for distinguishable colors
    state_colors = plt.cm.get_cmap('tab10', ns)

    # Create subplots for state and input trajectories
    fig, axs = plt.subplots(ns + 1, 1, figsize=(10, 3 * (ns + 1)), sharex=True)

    for i in range(ns):
        axs[i].plot(tt_hor, xstar[i, :], color=state_colors(i), linewidth=2, label=f'$x_{i+1}^{{*}}$')
        axs[i].plot(tt_hor, xref[i, :], '--', color='gray', linewidth=2, label=f'$x_{i+1}^{{ref}}$')
        axs[i].grid()
        if i <= 2:
            axs[i].set_ylabel(f'$x_{i+1}$ [rad]')
        else:
            axs[i].set_ylabel(f'$x_{i+1}$ [rad/s]')
        axs[i].legend()
        axs[i].set_title(f'Optimal state trajectory $x_{i+1}$ vs reference curve')

    # Input trajectory subplot
    axs[ns].plot(tt_hor, ustar[0, :], 'r', linewidth=2, label='$u^*$')
    axs[ns].plot(tt_hor, uref[0, :], '--', color='gray', linewidth=2, label='$u^{ref}$')
    axs[ns].grid()
    axs[ns].set_ylabel('$u$ [Nm]')
    axs[ns].set_xlabel('Time [s]')
    axs[ns].set_title('Optimal input trajectory vs reference curve')
    axs[ns].legend()

    fig.tight_layout()
    plt.show()



# Trajectory tracking plots
def traj_track_plot(t_hor, xreal, xdes, ureal, udes, ns):
    """
    Plots the real state and input trajectories for trajectory tracking vs optimal reference trajectories.

    Parameters:
    t_hor : numpy array
        Time horizon array.
    xreal : numpy array
        Real state trajectory (shape: ns x T).
    xdes : numpy array
        Desired state trajectory (shape: ns x T).
    ureal : numpy array
        Real input trajectory (shape: 1 x T).
    udes : numpy array
        Desired input trajectory (shape: 1 x T).
    ns : int
        Number of states.
    """
    # Set up the color palette for distinguishable colors
    state_colors = plt.cm.get_cmap('tab10', ns)

    # Create subplots for state and input trajectories
    fig, axs = plt.subplots(ns + 1, 1, figsize=(10, 3 * (ns + 1)), sharex=True)

    for i in range(ns):
        axs[i].plot(t_hor, xreal[i, :], color=state_colors(i), linewidth=2, label=f'$x_{i+1}^{{real}}$')
        axs[i].plot(t_hor, xdes[i, :], '--', color='gray', linewidth=2, label=f'$x_{i+1}^{{des}}$')
        axs[i].grid()
        if i <= 2:
            axs[i].set_ylabel(f'$x_{i+1}$ [rad]')
        else:
            axs[i].set_ylabel(f'$x_{i+1}$ [rad/s]')
        axs[i].legend()
        axs[i].set_title(f'State $x_{i+1}$ trajectory tracking', fontsize=10)

    # Input trajectory subplot
    axs[ns].plot(t_hor, ureal[0, :], 'r', linewidth=2, label='$u^{real}$')
    axs[ns].plot(t_hor, udes[0, :], '--', color='gray', linewidth=2, label='$u^{des}$')
    axs[ns].grid()
    axs[ns].set_ylabel('$u$ [Nm]')
    axs[ns].set_xlabel('Time [s]')
    axs[ns].set_title('Input trajectory tracking', fontsize=10)
    axs[ns].legend()

    fig.tight_layout()
    plt.show()


# Tracking error plots
def track_error_plot(t_hor, xreal, xdes, ureal, udes, ns):
    """
    Plots the state and input tracking errors for trajectory tracking.

    Parameters:
    t_hor : numpy array
        Time horizon array.
    xreal : numpy array
        Real state trajectory (shape: ns x T).
    xdes : numpy array
        Desired state trajectory (shape: ns x T).
    ureal : numpy array
        Real input trajectory (shape: 1 x T).
    udes : numpy array
        Desired input trajectory (shape: 1 x T).
    ns : int
        Number of states.
    """
    # Set up the color palette for distinguishable colors
    state_colors = plt.cm.get_cmap('tab10', ns)

    # Create subplots for state and input tracking errors
    fig, axs = plt.subplots(ns + 1, 1, figsize=(10, 3 * (ns + 1)), sharex=True)

    for i in range(ns):
        # Calculate state tracking error
        x_error = xreal[i, :] - xdes[i, :]
        
        axs[i].plot(t_hor, x_error, color=state_colors(i), linewidth=2, label=f'$e_{i+1} = x_{i+1}^{{real}} - x_{i+1}^{{des}}$')
        axs[i].axhline(0, color='gray', linestyle='--', linewidth=1)
        axs[i].grid()
        if i <= 2:
            axs[i].set_ylabel(f'$e_{i+1}$ [rad]')
        else:
            axs[i].set_ylabel(f'$e_{i+1}$ [rad/s]')
        axs[i].legend()
        axs[i].set_title(f'State $x_{i+1}$ tracking error', fontsize=10)

    # Input tracking error subplot
    u_error = ureal[0, :] - udes[0, :]
    axs[ns].plot(t_hor, u_error, 'r', linewidth=2, label='$e_u = u^{real} - u^{des}$')
    axs[ns].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[ns].grid()
    axs[ns].set_ylabel('$e_u$ [Nm]')
    axs[ns].set_xlabel('Time [s]')
    axs[ns].set_title('Input tracking error', fontsize=10)
    axs[ns].legend()

    fig.tight_layout()
    plt.show() 