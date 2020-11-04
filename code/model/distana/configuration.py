"""
This file contains the configuration for the kernel network
"""

#
# General configurations

SOURCE_PATH = "/home/karlbau/Documents/phd/weather_prediction/pytorch/wave_simulation/"

ARCHITECTURE_NAME = "distana"
MODEL_NAME = "tmp_model"

DATA_TYPE = "tmp_data/"
DATA_NOISE = 0.0  # 5e-5  # The noise that is added to the input data
P_ZERO_INPUT = 0.0  # Probability of feeding no input to a PK

DEVICE = "CPU"  # or "CPU" - for grid sizes > 25x25 the GPU version is faster

#
# Training parameters

SAVE_MODEL = True
CONTINUE_TRAINING = False

EPOCHS = 100
SEQ_LEN = 40  # 150
LEARNING_RATE = 0.001

#
# Testing parameters

TEACHER_FORCING_STEPS = 15
CLOSED_LOOP_STEPS = 135

#
# PK specific configurations

PK_ROWS = 16  # Rows of PKs
PK_COLS = 16  # Cols of PKs

PK_NEIGHBORS = 8
PK_DYN_IN_SIZE = 1  # the z-value of the wave field
PK_LAT_IN_SIZE = 1
PK_PRE_LAYER_SIZE = 4
PK_NUM_LSTM_CELLS = 16
PK_DYN_OUT_SIZE = 1  # Must be equal to PK_DYN_IN_SIZE
PK_LAT_OUT_SIZE = 1  # Must be equal to PK_LAT_IN_SIZE
