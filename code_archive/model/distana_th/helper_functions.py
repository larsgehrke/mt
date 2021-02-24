import numpy as np
import torch as th
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
import configuration as cfg


def determine_device():
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(th.cuda.get_device_name(0))
        print("Memory Usage:")
        print("\tAllocated:",
              round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(th.cuda.memory_cached(0) / 1024 ** 3, 1),
              "GB")
        print()
    return device


def save_model_to_file(model_src_path, cfg_file, epoch, epoch_errors_train,
                       epoch_errors_val, net):
    """
    This function writes the model weights along with the network configuration
    and current performance to file.
    :param model_src_path: The source path where the model will be saved to
    :param cfg_file: The configuration file
    :param epoch: The current epoch
    :param epoch_errors_train: The training epoch errors
    :param epoch_errors_val: The validation epoch errors,
    :param net: The actual model
    :return: Nothing
    """
    # print("\nSaving model (that is the network's weights) to file...")

    _model_save_path = model_src_path + "/" + cfg.MODEL_NAME + "/"
    if not os.path.exists(_model_save_path):
        os.makedirs(_model_save_path)

    # Save model weights to file
    th.save(net.state_dict(), _model_save_path + cfg.MODEL_NAME + ".pt")

    output_string = cfg_file + "\n#\n# Performance\n\n"

    output_string += "CURRENT_EPOCH = " + str(epoch) + "\n"
    output_string += "EPOCHS = " + str(cfg.EPOCHS) + "\n"
    output_string += "CURRENT_TRAINING_ERROR = " + \
                     str(epoch_errors_train[-1]) + "\n"
    output_string += "LOWEST_TRAINING_ERROR = " + \
                     str(min(epoch_errors_train)) + "\n"
    output_string += "CURRENT_VALIDATION_ERROR = " + \
                     str(epoch_errors_val[-1]) + "\n"
    output_string += "LOWEST_VALIDATION_ERROR = " + \
                     str(min(epoch_errors_val))

    # Save the configuration and current performance to file
    with open(_model_save_path + 'cfg_and_performance.txt', 'w') as _text_file:
        _text_file.write(output_string)
