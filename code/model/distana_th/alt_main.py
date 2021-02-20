import numpy as np
import numba
import sys
import glob
import time
import math
import torch as th
import torch.nn as nn

import configuration as cfg

import kernel_variables
import kernel_net

import helpers
from helper_functions import sprint

# JUST TRAINING FOR NOW

# GPU only Code
if cfg.DEVICE == "CPU":
    sys.exit("ERROR: CPU mode is enabled, but this is GPU code only.")
elif not th.cuda.is_available():
    sys.exit("ERROR: This is GPU code only, but Pytorch reports that CUDA is not available.")

# Print some information about the GPU
print(th.cuda.get_device_name(0))
print("Memory Usage:")
print("\tAllocated:",
      round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
print("\tCached:   ", round(th.cuda.memory_cached(0) / 1024 ** 3, 1),
      "GB")
print()

cfg.TRAINING = True

# Specify the paths for this script
# (Two or more physical lines are joined into logical lines using backslash character)
data_src_path = cfg.SOURCE_PATH + "data/" + cfg.DATA_TYPE
model_src_path = cfg.SOURCE_PATH + "model/" + \
                 cfg.ARCHITECTURE_NAME + "/saved_models/"

# Compute amount of PKs
amount_pks = cfg.PK_ROWS * cfg.PK_COLS

# Set up the parameter and tensor classes
params = kernel_variables.KernelParameters(
    amount_pks=amount_pks,
    device=th.device("cuda")
)
tensors = kernel_variables.KernelTensors(params=params)

# Initialize and set up the kernel network
# build adjacent matrix and configure all linkages
net = kernel_net.KernelNetwork(
    params=params,
    tensors=tensors
)

# Count number of trainable parameters
pytorch_total_params = sum(
    p.numel() for p in net.parameters() if p.requires_grad
)
print("Trainable model parameters:", pytorch_total_params)


# Set up the optimizer and the criterion (loss)
optimizer = th.optim.Adam(list(net.parameters()), lr=cfg.LEARNING_RATE) #TODO: Error: net.parameters() ist empty: th.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
criterion = nn.MSELoss()

net.pk_net.to("cuda")

# Set up lists to save and store the epoch errors
epoch_errors_train = []
epoch_errors_val = []

best_train = np.infty
best_val = np.infty

# Get the training and validation file names
train_data_filenames = np.sort(glob.glob(data_src_path + 'train/*'))#[:1]
val_data_filenames = np.sort(glob.glob(data_src_path + 'val/*'))

if len(train_data_filenames) == 0:
    raise Exception('Could not find training data in '+ str(data_src_path) + 'train/*')
if len(val_data_filenames) == 0:
    raise Exception('Could not find validation data in '+ str(data_src_path) + 'val/*')

# Optional: Load Data from file

"""
TRAINING
"""

training_start_time = time.time()

# Start the training and iterate over all epochs
for epoch in range(cfg.EPOCHS):
    epoch_start_time = time.time()

    # Shuffle training data
    np.random.shuffle(train_data_filenames)

    # save epoch errors
    batch_errors = []

    # Calculate number of iterations
    amount_batches = math.ceil(len(train_data_filenames)/cfg.BATCH_SIZE)

    for batch_iter in range(amount_batches):

        # Training
        # Forward pass
        # backward pass
        #training error (mean squared error)
        mse = helpers.train_batch(
            net = net,
            data_filenames = train_data_filenames,
            criterion=criterion,
            optimizer=optimizer,
            batch_iter= batch_iter,
            params = params,
            tensors = tensors
                )

        batch_errors.append(mse) # mse.item()
        # mse tupel 3 x (8, 40, 256, 1)

    epoch_errors_train.append(0)#np.mean(batch_errors))

    # Create a plus or minus sign for the training error
    train_sign = "(-)"
    if epoch_errors_train[-1] < best_train:
        train_sign = "(+)"
        best_train = epoch_errors_train[-1]

    # Compute validation error
    # . . .
    # mse = helpers.validate(
    #         net = net,
    #         data_filenames = val_data_filenames,
    #         criterion=criterion
    #             )

    # epoch_errors_val.append(mse) # mse.item()

    # Create a plus or minus sign for the validation error
    # val_sign = "(-)"
    # if epoch_errors_val[-1] < best_val:
    #     best_val = epoch_errors_val[-1]
    #     val_sign = "(+)"


    # Print progress to the console with nice formatting
    print('Epoch ' + str(epoch + 1).zfill(int(np.log10(cfg.EPOCHS)) + 1)
          + '/' + str(cfg.EPOCHS) + ' took '
          + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')
          + ' seconds.'
          # + ' seconds.\t\tAverage epoch training error: ' + train_sign
          # + str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')
          # + '\t\tValidation error: ' + val_sign
          # + str(np.round(epoch_errors_val[-1], 10)).ljust(12, ' ')
          )

program_duration = np.round(time.time() - training_start_time, 2)
print('\nTraining took ' + str(program_duration) + ' seconds.\n\n')
print("Done!")































