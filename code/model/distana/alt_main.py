import numpy as np
import numba
import sys
import configuration as cfg
import torch as th

# JUST TRAINING FOR NOW

# GPU only Code
if cfg.DEVICE == "CPU":
    sys.exit("ERROR: CPU mode is enabled, but this is GPU only code.")
elif not th.cuda.is_available():
    sys.exit("ERROR: This is GPU only code, but Pytorch says CUDA is not available.")

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




