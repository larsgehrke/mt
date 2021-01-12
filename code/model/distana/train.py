"""
In this script, a neural network model is created to simulate the propagation
of a wave by using the PyTorch library.

To model the waves, a grid of Prediction Kernels (PKs) is initialized that are
laterally connected to propagate spatial information. Temporal information is
processed by the PKs that consist of some feed forward layers and a long-short
term memory (LSTM) core. A crucial component of this model is that all PKs share
weights and thus all realize the same computations, just differently
parameterized by dynamic and optional static inputs
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import time
import matplotlib.pyplot as plt
from threading import Thread
import kernel_variables
import kernel_net
import configuration as cfg
import helper_functions as helpers

# Hide the GPU(s) in case the user specified to use the CPU in the config file
if cfg.DEVICE == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_training():
    # Set a globally reachable boolean in the config file for training
    cfg.TRAINING = True

    # Read the configuration file to be able to save it later if desired
    if cfg.SAVE_MODEL:
        with open('configuration.py', 'r') as f:
            cfg_file = f.read()

    # Print some information to console
    print("Architecture name:", cfg.ARCHITECTURE_NAME)
    print("Model name:", cfg.MODEL_NAME)

    # Specify the paths for this script
    # (Two or more physical lines may be joined into logical lines using backslash character)
    data_src_path = cfg.SOURCE_PATH + "data/" + cfg.DATA_TYPE
    model_src_path = cfg.SOURCE_PATH + "model/" + \
                     cfg.ARCHITECTURE_NAME + "/saved_models/"

    # Set device on GPU if specified in the configuration file, else CPU
    # (just depends on th.cuda.is_available())
    device = helpers.determine_device()

    # Compute batch size for the PKs (every PK is processed in a separate sample of the batch
    # to parallelize computation) 
    # TRICK: Batch size used for parallelization
    # Batch size depends on amount of PKs
    pk_batches = cfg.PK_ROWS * cfg.PK_COLS

    # Set up the parameter and tensor classes
    params = kernel_variables.KernelParameters(
        pk_batches=pk_batches,
        device=device
    )
    tensors = kernel_variables.KernelTensors(params=params)

    # Initialize and set up the kernel network
    net = kernel_net.KernelNetwork(
        params=params,
        tensors=tensors
    )

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    epoch_errors_val = []

    best_train = np.infty
    best_val = np.infty

    #
    # Get the training and validation file names
    train_data_filenames = np.sort(glob.glob(data_src_path + 'train/*'))#[:1]
    val_data_filenames = np.sort(glob.glob(data_src_path + 'val/*'))

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if cfg.CONTINUE_TRAINING:
        print('Restoring model (that is the network\'s weights) from file...')
        net.load_state_dict(th.load(model_src_path + "/" + cfg.MODEL_NAME + "/"
                                    + cfg.MODEL_NAME + ".pt"))
        net.eval()

    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(cfg.EPOCHS):

        epoch_start_time = time.time()

        # Shuffle the training_data_filenames to have variable training data
        # and define the number of iterations that shall be performed during
        # one epoch
        np.random.shuffle(train_data_filenames)

        sequence_errors = []

        # Iterate over all training iterations and evaluate the network
        for train_iter in range(len(train_data_filenames)):
        #for train_iter in range(100):

            # Evaluate and train the network for the given training data
            mse, _, _, _ = helpers.evaluate(
                net=net,
                data_filenames=train_data_filenames,
                params=params,
                tensors=tensors,
                pk_batches=pk_batches,
                criterion=criterion,
                optimizer=optimizer,
                _iter=train_iter
            )

            sequence_errors.append(mse.item())

        epoch_errors_train.append(np.mean(sequence_errors))

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]

        #
        # Compute validation error

        # Evaluate and validate the network for the given validation data
        mse, _, _, _ = helpers.evaluate(
            net=net,
            data_filenames=val_data_filenames,
            params=params,
            tensors=tensors,
            pk_batches=pk_batches,
            criterion=criterion,
        )

        epoch_errors_val.append(mse.item())

        # Save the model to file (if desired)
        if cfg.SAVE_MODEL and mse.item() < best_val:
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=model_src_path,
                cfg_file=cfg_file,
                epoch=epoch,
                epoch_errors_train=epoch_errors_train,
                epoch_errors_val=epoch_errors_val,
                net=net))
            thread.start()

        # Create a plus or minus sign for the validation error
        val_sign = "(-)"
        if epoch_errors_val[-1] < best_val:
            best_val = epoch_errors_val[-1]
            val_sign = "(+)"

        #
        # Print progress to the console
        print('Epoch ' + str(epoch + 1).zfill(int(np.log10(cfg.EPOCHS)) + 1)
              + '/' + str(cfg.EPOCHS) + ' took '
              + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')
              + ' seconds.\t\tAverage epoch training error: ' + train_sign
              + str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')
              + '\t\tValidation error: ' + val_sign
              + str(np.round(epoch_errors_val[-1], 10)).ljust(12, ' '))

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')

print("Done!")
