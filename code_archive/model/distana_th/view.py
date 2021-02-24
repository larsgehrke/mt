import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
import configuration as cfg


def plot_kernel_activity(ax, label, net_out, net_in=None, make_legend=False):
    """
    This function displays the wave activity of a single kernel.
    :param ax: The plot where the activity shall be displayed in
    :param label: The label for the wave (ground truth)
    :param net_out: The network output
    :param net_in: The network input
    :param make_legend: Boolean that indicates weather a legend shall be created
    """

    central_kernel = (cfg.PK_ROWS * cfg.PK_COLS) // 2
    central_kernel = 20

    if net_in is not None:
        ax.plot(range(len(net_in)), net_in[:, central_kernel, 0],
                label='Network input', color='green')
    ax.plot(range(len(label)), label[:, central_kernel, 0],
            label='Target', color='deepskyblue')
    ax.plot(range(len(net_out)), net_out[:, central_kernel, 0],
            label='Network output', color='red', linestyle='dashed')
    # if net_in is None:
    yticks = ax.get_yticks()[1:-1]
    ax.plot(np.ones(len(yticks)) * cfg.TEACHER_FORCING_STEPS, yticks,
            color='white', linestyle='dotted',
            label='End of teacher forcing')
    if make_legend:
        ax.legend()


def animate_2d_wave(net_label, net_outputs, net_inputs=None):
    """
    This function visualizes the spatio-temporally expanding wave
    :param net_label: The corresponding labels
    :param net_outputs: The network output
    :param net_inputs: The network inputs
    :return: The animated plot of the 2d wave
    """

    num_axes = 2

    # Bring the data into a format that can be displayed as heatmap
    data = np.reshape(net_outputs, [len(net_outputs),
                                    cfg.PK_ROWS,
                                    cfg.PK_COLS,
                                    len(net_outputs[0, 0])])
    net_label = np.reshape(net_label, [len(net_label),
                                       cfg.PK_ROWS,
                                       cfg.PK_COLS,
                                       len(net_label[0, 0])])

    if net_inputs is not None:
        net_inputs = np.reshape(net_inputs, [len(net_inputs),
                                             cfg.PK_ROWS,
                                             cfg.PK_COLS,
                                             len(net_inputs[0, 0])])
        num_axes = 3

    # Define a grid size that shall be visualized
    gs1 = 0
    gs2 = cfg.PK_ROWS

    # First set up the figure, the axis, and the plot element we want to
    # animate
    fig, axes = plt.subplots(1, num_axes, figsize=[6*num_axes, 6], dpi=100)
    im1 = axes[0].imshow(net_label[0, gs1:gs2, gs1:gs2, 0], vmin=-0.8, vmax=0.8,
                         cmap='Blues')

    # Visualize the obstacle if there is one
    txt1 = axes[0].text(0, axes[0].get_yticks()[0], 't = 0', fontsize=20,
                        color='white')
    axes[0].set_title("Network Output")

    # In the subfigure on the right hand side, visualize the true data
    im2 = axes[1].imshow(net_label[0, gs1:gs2, gs1:gs2, 0], vmin=-0.8, vmax=0.8,
                         cmap='Blues')
    axes[1].set_title("Ground Truth")

    im3 = None
    if net_inputs is not None:
        im3 = axes[2].imshow(net_inputs[0, gs1:gs2, gs1:gs2, 0], vmin=-0.8,
                             vmax=0.8, cmap="Blues")
        axes[2].set_title("Network Input")

    anim = animation.FuncAnimation(fig, animate, frames=len(data),
                                   fargs=(cfg.TEACHER_FORCING_STEPS, data, im1,
                                          im2, im3, txt1, gs1, gs2, net_label,
                                          net_inputs),
                                   interval=1)

    return anim


def animate(_i, _teacher_forcing_steps, _data, _im1, _im2, _im3, _txt1, _gs1,
            _gs2, _net_label, _net_inputs):

    # Pause the simulation briefly when switching from teacher forcing to
    # closed loop prediction
    if _i == _teacher_forcing_steps:
        time.sleep(1.0)
    elif _i < 150:
        time.sleep(0.05)

    # Set the pixel values of the image to the data of timestep _i
    _im1.set_array(_data[_i, _gs1:_gs2, _gs1:_gs2, 0])
    if _i < len(_net_label) - 1:
        _im2.set_array(_net_label[_i, _gs1:_gs2, _gs1:_gs2, 0])
        if _im3 is not None:
            _im3.set_array(_net_inputs[_i, _gs1:_gs2, _gs1:_gs2, 0])

    # Display the current timestep in text form in the plot
    if _i < _teacher_forcing_steps:
        _txt1.set_text('Teacher forcing, t = ' + str(_i))
    else:
        _txt1.set_text('Closed loop prediction, t = ' + str(_i))

    return _im1, _im2, _im3
