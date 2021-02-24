description = '''

 This is the general testing script for different versions of
 DISTANA (Karlbauer et al. 2019).

 The Graph Neural Network DISTANA is created to simulate the propagation
 of a wave by using the PyTorch library.

 To model the waves, a grid of Prediction Kernels (PKs) is initialized that are
 laterally connected to propagate spatial information. Temporal information is
 processed by the PKs that consist of some feed forward layers and a long-short
 term memory (LSTM) core. A crucial component of this model is that all PKs share
 weights and thus all realize the same computations, just differently
 parameterized by dynamic and optional static input

'''

from config.distana_params import DISTANAParams
from tools.persistence import FileManager


param_manager = DISTANAParams(FileManager(), description)
# Load parameters from file
params = param_manager.parse_params(is_training = False)



