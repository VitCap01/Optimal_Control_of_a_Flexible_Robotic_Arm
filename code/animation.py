import pygame
import numpy as np

def animation(x, xdes):

    """
    This function creates an animation of the robot using Pygame. The animation includes 
    interactive buttons to control the simulation's state (e.g., "UP" to move the pendulum up, 
    "RESTART" to reset the simulation). The pendulum dynamics are visualized in real-time, 
    and the function also compares the actual trajectory with a desired trajectory.

    Inputs:
    - x: 2D numpy array, actual state trajectory.
    - xdes: 2D numpy array, desired state trajectory
    """

    # Initialize pygame
    pygame.init()

    # Window settings
    WIDTH, HEIGHT = 600, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum")
    clock = pygame.time.Clock()

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GRAY = (50, 50, 50)
    GREEN = (0, 255, 0)

    # Pendulum center
    CENTER = (WIDTH // 2, HEIGHT // 2)

    # Pendulum variables (example values)
    l1, l2 = 2, 2  # Lengths of the pendulums (in units)
    l1 *= 50  # Length of the first pendulum (pixels)
    l2 *= 50  # Length of the second pendulum (pixels)
    r1, r2 = l1 / 2, l2 / 2
    t = 0

    # Function to draw a button
    def draw_button(screen, rect, color, text, text_color):
        pygame.draw.rect(screen, color, rect)
        font = pygame.font.Font(None, 36)
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    def is_near_multiple(a, b, tolerance=0.01):
        difference = abs(np.cos(a) - np.cos(b))  # Difference between the number and the nearest multiple
        return difference <= tolerance

    # Button positions
    up_button = pygame.Rect(50, 500, 150, 50)  # "Start" button
    restart_button = pygame.Rect(400, 500, 150, 50)  # "Reset" button

    # Variable to control whether the simulation is running
    is_running = False
    is_up = False
    is_down = True

    # Drawing on pygame
    screen.fill(BLACK)  # Clear the screen
    t = 0
    incremento = 0
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Detect mouse clicks on buttons
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                incremento = 0
                if is_running:
                    if up_button.collidepoint(mouse_pos) and is_down:
                        t = 0  # Reset the time
                    elif restart_button.collidepoint(mouse_pos) and is_up:
                        t = 0  # Reset the time
                else:
                    if up_button.collidepoint(mouse_pos) and is_down:
                        is_running = not is_running  # Start or pause the simulation
                        if t > 4:
                            t = t - 3  # Reset the time
            else:
                incremento = 3
        # Drawing on pygame
        screen.fill(BLACK)  # Clear the screen

        # Calculate the positions of the pendulums (if the simulation is running)
        if is_running and t < x.shape[1]:  # Ensure we don't exceed the time dimension
            x1 = CENTER[0] + l1 * np.sin(x[0, t])
            y1 = CENTER[1] + l1 * np.cos(x[0, t])
            x2 = x1 + l2 * np.sin(x[0, t] + x[1, t])
            y2 = y1 + l2 * np.cos(x[0, t] + x[1, t])

            x10 = CENTER[0] + l1 * np.sin(xdes[0, t])
            y10 = CENTER[1] + l1 * np.cos(xdes[0, t])
            x20 = x10 + l2 * np.sin(xdes[0, t] + xdes[1, t])
            y20 = y10 + l2 * np.cos(xdes[0, t] + xdes[1, t])
        else:
            # Keep the current position when the simulation is paused
            x1 = CENTER[0] + l1 * np.sin(x[0, 0])
            y1 = CENTER[1] + l1 * np.cos(x[0, 0])
            x2 = x1 + l2 * np.sin(0)
            y2 = y1 + l2 * np.cos(0)

            x10 = CENTER[0] + l1 * np.sin(0)
            y10 = CENTER[1] + l1 * np.cos(0)
            x20 = x10 + l2 * np.sin(0)
            y20 = y10 + l2 * np.cos(0)

        # Draw the rods and pendulums
        pygame.draw.line(screen, GRAY, CENTER, (x10, y10), 1)  # First rod
        pygame.draw.line(screen, GRAY, (x10, y10), (x20, y20), 1)  # Second rod
        pygame.draw.circle(screen, GRAY, (int(x10), int(y10)), 7)  # First pendulum
        pygame.draw.circle(screen, GRAY, (int(x20), int(y20)), 7)  # Second pendulum

        # Draw the rods and pendulums
        pygame.draw.line(screen, WHITE, CENTER, (x1, y1), 1)  # First rod
        pygame.draw.line(screen, WHITE, (x1, y1), (x2, y2), 1)  # Second rod
        pygame.draw.circle(screen, RED, (int(x1), int(y1)), 7)  # First pendulum
        pygame.draw.circle(screen, BLUE, (int(x2), int(y2)), 7)  # Second pendulum

        # Draw the buttons
        draw_button(screen, up_button, GREEN if is_down else GRAY, "UP", WHITE)
        draw_button(screen, restart_button, GREEN if is_up else GRAY, "RESTART", WHITE)

        pygame.display.flip()

        if is_running and t < x.shape[1]:  # Ensure we don't exceed the time dimension
            if is_near_multiple(x[0, t], np.pi) and is_near_multiple(x[1, t], 0) and t / x.shape[1] > 0.55:
                is_up = True
            else:
                is_up = False
            if is_near_multiple(x[0, t], 0) and is_near_multiple(x[1, t], np.pi) and t / x.shape[1] > 0.45:
                is_down = True
            else:
                is_down = False

        # Handle the update of time
        if is_running:
            if not t >= x.shape[1] - 3 or t < 10:
                t += incremento
            else:
                t = t

    pygame.quit()



