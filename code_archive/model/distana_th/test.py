import numpy as np
import torch as th
import time
import glob
import os
import matplotlib.pyplot as plt
import kernel_variables
import kernel_net
import configuration as cfg
import debug as helpers

# Hide the GPU(s) in case the user specified to use the CPU in the config file
if cfg.DEVICE == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_testing():
    # Set a globally reachable boolean in the config file for testing
    cfg.TRAINING = False

    # Specify the paths for this script
    data_src_path = cfg.SOURCE_PATH + "data/" + cfg.DATA_TYPE
    model_src_path = cfg.SOURCE_PATH + "model/" + \
                     cfg.ARCHITECTURE_NAME + "/saved_models/"

    # setting device on GPU if available, else CPU
    device = helpers.determine_device()

    # Compute batch size for PKs and TKs
    pk_batches = cfg.PK_ROWS * cfg.PK_COLS

    # Set up the parameter and tensor classes
    params = kernel_variables.KernelParameters(
        pk_batches=pk_batches,
        device=device
    )
    # tensors1 = kernel_variables.KernelTensors(_params=params)
    tensors = kernel_variables.KernelTensors(params=params)

    # Initialize and set up the kernel network
    net = kernel_net.KernelNetwork(
        params=params,
        tensors=tensors
    )

    # Restore the network by loading the weights saved in the .pt file
    print('Restoring model (that is the network\'s weights) from file...')
    net.load_state_dict(th.load(model_src_path + "/" + cfg.MODEL_NAME + "/" +
                                cfg.MODEL_NAME + ".pt",
                                map_location=params.device))
    net.eval()

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    """
    TESTING
    """

    plt.style.use('dark_background')

    #
    # Set up the feed dictionary for the test iteration
    test_data_filenames = glob.glob(data_src_path + 'test/*')[:10]

    # Test statistics
    time_list = []
    accuracy_list = []

    x = u'1'
    curr_idx = 0
    while x == u'1':

        time_start = time.time()

        # Evaluate the network for the given test data
        _, net_input, net_label, net_outputs = helpers.evaluate(
            net=net,
            data_filenames=test_data_filenames,
            params=params,
            tensors=tensors,
            pk_batches=pk_batches,
            _iter=curr_idx,
            testing=True
        )

        forward_pass_duration = time.time() - time_start
        print("\tForward pass took:", forward_pass_duration, "seconds.")

        net_outputs = net_outputs.detach().numpy()

        # Plot the wave activity
        fig, axes = plt.subplots(2, 2, figsize=[10, 10], sharex="all")
        for i in range(2):
            for j in range(2):
                make_legend = True if (i == 0 and j == 0) else False
                helpers.plot_kernel_activity(
                    ax=axes[i, j],
                    label=net_label,
                    net_out=net_outputs,
                    net_in=net_input,
                    make_legend=make_legend
                )
        fig.suptitle('Model ' + cfg.MODEL_NAME, fontsize=12)
        plt.show()

        # Visualize and animate the propagation of the 2d wave
        anim = helpers.animate_2d_wave(net_label, net_outputs, net_input)
        #anim = helpers.animate_2d_wave(net_label, net_outputs)
        plt.show()

        # Retrieve user input to continue or quit the testing
        x = input("Press 1 to see another example, anything else to quit.")
        curr_idx += 1

        # Append the test statistics for this sequence to the appropriate lists
        time_list.append(forward_pass_duration)
        #accuracy_list.append(test_error_mean)

    print("Average forward pass duration:", np.mean(time_list),
          " +-", np.std(time_list))
    #print("Average accuracy:", np.mean(accuracy_list),
    #      " +-", np.std(accuracy_list))

    print('Done')
