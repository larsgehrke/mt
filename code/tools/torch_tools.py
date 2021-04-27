'''

    Helper functions for the handling of PyTorch specific tasks.

'''

import torch as th
from threading import Thread
import os

def determine_device(use_cuda: bool) -> str:
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() and use_cuda else "cpu")
    #print("Using device:", device)
    #print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(th.cuda.get_device_name(0))
        # Deactivated the following lines, because the allocated memory was always 0
        # print("Memory Usage:")
        # print("\tAllocated:",
        #       round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        # print("\tCached:   ", round(th.cuda.memory_reserved(0) / 1024 ** 3, 1),
        #       "GB")
        print()

    return device

def load_model(params: dict):
    '''
    Load trained PyTorch model from file.
    :return loaded PyTorch model
    '''
    #print('Restoring model (that is the network\'s weights) from file...')
    net = th.load(os.path.join(params['model_folder'], params['model_name'] + ".pt"),
            map_location=params['device'])

    return net

class Saver():
    '''
        Save trained PyTorch model to file.
    '''
    def __init__(self, epochs: int, model_src_path: str, 
        model_name: str, cfg_file: str, net):
        '''
        Initialization of the Saver class.
        :param epochs: number of epochs
        :param model_src_path: path to folder where the model should be saved
        :param model_name: model/version name 
        :param cfg_file: complete configuration information
        :param net: the PyTorch network that should be saved
        '''
        
        self.epochs = epochs
        self.model_save_path=model_src_path
        self.cfg_file=cfg_file
        self.net=net
        self.model_name = model_name


    def __call__(self, epoch: int, epoch_errors_train: list, epoch_errors_val: list):
        '''
        This class can be called to save a model to file
        after the results of a specific epoch were satisfying.
        :param epoch: The last epoch index 
        :param epoch_errors_train: the training errors of the last epoch
        :param epoch_errors_val: the validation errors of the last epoch
        '''

        # Start a separate thread to save the model
        thread = Thread(target=self.save_model_to_file(epoch, 
                        epoch_errors_train,epoch_errors_val))
        thread.start()

    def save_model_to_file(self, epoch: int, epoch_errors_train: list,
                       epoch_errors_val: list):
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
    
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Save model weights to file
        th.save(self.net.state_dict(), self.model_save_path + self.model_name + ".pt")

        output_string = self.cfg_file + "\n#\n# Performance\n\n"

        output_string += "CURRENT_EPOCH = " + str(epoch) + "\n"
        output_string += "EPOCHS = " + str(self.epochs) + "\n"
        output_string += "CURRENT_TRAINING_ERROR = " + \
                         str(epoch_errors_train[-1]) + "\n"
        output_string += "LOWEST_TRAINING_ERROR = " + \
                         str(min(epoch_errors_train)) + "\n"
        output_string += "CURRENT_VALIDATION_ERROR = " + \
                         str(epoch_errors_val[-1]) + "\n"
        output_string += "LOWEST_VALIDATION_ERROR = " + \
                         str(min(epoch_errors_val))

        # Save the configuration and current performance to file
        with open(self.model_save_path + 'cfg_and_performance.txt', 'w') as _text_file:
            _text_file.write(output_string)





