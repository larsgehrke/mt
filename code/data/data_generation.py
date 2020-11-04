"""
In this file we generate wave data to train a network. Waves are generate by
using the wave equation and some fading properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

#
# GLOBAL VARIABLES

#
# Simulation parameters
dt = 0.1  # Temporal step size
dx = 1  # Step size in x-direction
dy = 1  # Step size in y-direction
time_steps = 150  # Number of simulation steps

#
# Field parameters
width = 16  # Width of the simulated field in pixels
height = 16  # Height of the simulated field in pixels

#
# Wave parameters
wave_width_x = 0.5  # Width of the wave in x-direction
wave_width_y = 0.5  # Width of the wave in y-direction
amplitude = 0.34  # Amplitude of the wave
velocity = 3.0  # The velocity of the wave
waves = 1  # The number of waves that propagate simultaneously in one sequence
damp = 1.0  # How much a wave is dampened (decaying over time)

variable_velocity = False  # If true, waves of different speed are generated
variable_amplitude = False  # If true, waves have different amplitudes
velocities = [0.5, 5.0]  # The min and max values for the wave speed
amplitudes = [0.1, 0.34]  # The min and max values for the wave amplitude

#
# Other parameters
create_n_files = 100  # The number of data files that shall be created
data_set = 'val'  # Can be 'test', 'train' or 'val'

save_data = False  # Shall the generated data be saved to file
visualize = True  # Create a plot and animation of the wave
threshold_empty_data = 0.00  # In 1%, an empty data set (only zeros) is created

data_name = 'tmp_data'

velocity_ary = np.ones((width, height)) * 3
# velocity_ary[:, 20:] = 16


#
# FUNCTIONS

def f(_x, _y, _varx, _vary, _a):
    """
    Function to set the initial activity of the field. We use the Gaussian bell
    curve to initialize the field smoothly.
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _varx: The variance in x-direction
    :param _vary: The variance in y-direction
    :param _a: The amplitude of the wave
    :return: The initial activity at (x, y)
    """
    x_part = ((_x - start_pt[0])**2) / (2 * _varx)
    y_part = ((_y - start_pt[1])**2) / (2 * _vary)
    return _a * np.exp(-(x_part + y_part))


def g(_x, _y, _varx, _vary, _a):
    """
    Function to determine the changes over time in the field
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _varx: The variance in x-direction
    :param _vary: The variance in y-direction
    :param _a: The amplitude of the wave
    :return: The changes over time in the field at (x, y)
    """
    # x_part = _x * f(_x, _y, _varx, _vary, _a)
    # y_part = _y * f(_x, _y, _varx, _vary, _a)
    # return (x_part + y_part) / 2.
    return 0.0


def u(_t, _x, _y):
    """
    Function to calculate the field activity in time step t at (x, y)
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :param _c: The wave velocity
    :return: The field activity at position x, y in time step t.
    """

    # Compute changes in x- and y-direction
    dxxu = dxx_u(_t, _x, _y)
    dyyu = dyy_u(_t, _x, _y)

    # Get the activity at x and y in time step t
    u_t = field[t, _x, _y]

    # Catch initial condition, where there is no value of the field at time step
    # (t-1) yet
    if _t == 0:
        u_t_1 = dt_u(_x, _y)
        # u_t_1 = 0.0
    else:
        u_t_1 = field[_t - 1, _x, _y]

    velocity = velocity_ary[_x, _y]

    # Incorporate the changes in x- and y-direction and return the activity
    return damp * (((velocity**2) * (dt ** 2)) * (dxxu + dyyu) + 2 * u_t - u_t_1)


def dxx_u(_t, _x, _y):
    """
    The second derivative of u to x. Computes the lateral activity change in
    x-direction.
    Neuman Boundary conditions to prevent waves from reflecting at edges are
    taken from https://12000.org/my_notes/neumman_BC/Neumman_BC.htm
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: Field activity in t, considering changes in x-direction
    """

    # (Neuman) Boundary condition at left end of the field
    if _x == 0:
        dx_left = 0.0

        # g_xy = g(_x, _y, wave_width_x, wave_width_y, amplitude)
        # u_right = field[_t, _x + 1, _y]
        #
        # if _y == 0:
        #     u_top = 0.0
        # else:
        #     u_top = field[_t, _x, _y - 1]
        #
        # if _y == height - 1:
        #     u_bot = 0.0
        # else:
        #     u_bot = field[_t, _x, _y + 1]
        #
        # u_xy = field[_t, _x, _y]
        #
        # dx_left = (1/4) * (2*u_right - 2*dx*g_xy + u_top + u_bot + dt**2*u_xy)
    else:
        dx_left = field[_t, _x - dx, _y]

    # Boundary condition at right end of the field
    if _x == width - 1:
        dx_right = 0.0
    else:
        dx_right = field[_t, _x + dx, _y]

    # Calculate change in x-direction and return it
    ut_dx = dx_right - 2*field[_t, _x, _y] + dx_left

    return ut_dx / np.square(dx)


def dyy_u(_t, _x, _y):
    """
    The second derivative of u to y. Computes the lateral activity change in
    y-direction.
    :param _t: The current time step
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: Field activity in t, considering changes in y-direction
    """

    # Boundary condition at top end of the field
    if _y == 0:
        dy_above = 0.0
    else:
        dy_above = field[_t, _x, _y - dy]

    # Boundary condition at bottom end of the field
    if _y == height - 1:
        dy_below = 0.0
    else:
        dy_below = field[_t, _x, _y + dy]

    # Calculate change in y-direction and return it
    ut_dy = dy_below - 2*field[_t, _x, _y] + dy_above

    return ut_dy / np.square(dy)


def dt_u(_x, _y):
    """
    First derivative of u to t, only required in the very first time step to
    compute u(-dt, x, y).
    :param _x: The x-coordinate of the field (j, running over width)
    :param _y: The y-coordinate of the field (i, running over height))
    :return: The value of the field at (t-1), x, y
    """
    return field[1, _x, _y] - 2*dt*g(_x, _y, wave_width_x, wave_width_y,
                                     amplitude)


def animate(_t):
    im.set_array(field[_t, :, :])
    return im


#
# SCRIPT

for file_no in range(create_n_files):

    # If desired, choose a variable velocity for each sequence
    if variable_velocity:
        velocity = np.random.uniform(low=velocities[0], high=velocities[1])

    # If desired, choose a variable amplitude for each sequence
    if variable_amplitude:
        amplitude = np.random.uniform(low=amplitudes[0], high=amplitudes[1])

    # Print a progress statement to the console for every 20 generated files
    if (file_no + 1) % 20 == 0:
        print("Generating file " + str(file_no + 1) + "/" + str(create_n_files))

    # Initialize the wave field as two-dimensional zero-array
    field = np.zeros([time_steps, width, height])

    for wave in range(waves):
        # Generate a random point in the field where the impulse will be
        # initialized
        start_pt = np.random.randint(0, width, 2)

        # Compute the initial field activity by applying a 2D gaussian around
        # the start point
        for x in range(width):
            for y in range(height):
                field[0, x, y] += f(x, y, wave_width_x, wave_width_y, amplitude)

    # Iterate over all time steps to compute the activity at each position in
    # the grid over all time steps
    for t in range(time_steps - 1):

        # print("Running time step " + str(t) + "/" + str(time_steps))

        # Iterate over all values in the field and update them
        for x in range(width):
            for y in range(height):
                field[t + 1, x, y] = u(_t=t, _x=x, _y=y)

        # Normalize the field activities to be at most 1 (or -1)
        # print(np.max(np.abs(field)))
        # field = field / np.max(np.abs(field))

    if visualize:
        # plt.style.use('dark_background')
        # Plot the wave activity at one position
        fig, ax = plt.subplots(1, 1, figsize=[4, 3])
        ax.plot(range(time_steps), field[:, 5, 5])
        ax.set_xlabel("Time")
        ax.set_ylabel("Wave amplitude")
        plt.tight_layout()
        plt.show()

        # Animate the overall wave
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        im = ax.imshow(field[0, :, :], vmin=-0.6, vmax=0.6, cmap='Blues')
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames=time_steps,
                                       interval=200)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Draw a random value between 0 and 1
    rand_value = np.random.uniform(0, 1)

    # If the random value is smaller than the threshold, set the data on zeros
    if rand_value < threshold_empty_data:
        field = np.zeros_like(field)

    if save_data:

        dst_path = data_name + "/"

        # Check whether the directory to save the data exists and create it if
        # not
        if not os.path.exists(dst_path + data_set):
            os.makedirs(dst_path + data_set)

        # Define the impulse array, consisting of only zeros except for the cell
        # one time step before the initialization of the impulse. This cell is
        # one in this very time step
        impulse = np.zeros_like(field)
        impulse[0, start_pt[0], start_pt[1]] = 1.0

        # Concatenate the field data and the impulse data
        dat_save = np.concatenate([np.expand_dims(field, axis=1),
                                   np.expand_dims(impulse, axis=1)],
                                  axis=1)

        # Write the data to file
        np.save(dst_path + data_set + '/' + data_set
                + '_' + str(file_no).zfill(5), dat_save)
