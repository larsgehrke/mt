import numpy as np
import numba
import sys
import configuration as cfg
import torch as th

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
    pk_batches=pk_batches,
    device=device
)
tensors = kernel_variables.KernelTensors(params=params)

# Initialize and set up the kernel network
# build adjacent matrix and configure all linkages
net = kernel_net.KernelNetwork(
    params=params,
    tensors=tensors
)

# Set up the optimizer and the criterion (loss)
optimizer = th.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
criterion = nn.MSELoss()

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

start_0 = time.time()

# Start the training and iterate over all epochs
for epoch in range(cfg.EPOCHS):
    print(epoch)































