description = '''

 This is the general training script for different versions of
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
import os
import time
import numpy as np
import torch as th 

from model.distana import DISTANA
from config.distana_params import DISTANAParams
from config.params import Params
from tools.persistence import FileManager
from tools.persistence import get_data_filenames
from tools.supervisor import TrainSupervisor
import tools.torch_tools as th_tools




def run_training(params):

    # Create and set up the network
    distana = DISTANA(params)
    
    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(distana.net().parameters(), lr=params['learning_rate'])
    criterion = th.nn.MSELoss() 

    # Get the training and validation data
    train_data_files = get_data_filenames(os.path.join(params['data_folder'],'train',''))
    val_data_files = get_data_filenames(os.path.join(params['data_folder'],'val',''))

    # Set the training parameters for the model 
    # and get the amount of iterations for one epoch
    amount_iter = distana.set_training(train_data_files, optimizer, criterion)

    model_saver = None
    if params["save_model"]:
        model_saver = th_tools.Saver(epochs = params["epochs"],
            model_src_path=params['model_folder'], 
            version_name=params["version_name"], 
            cfg_file=Params.to_string(params), 
            net=distana.net())
    
    supervisor = TrainSupervisor(params['epochs'], model_saver)

    if params["continue_training"]:
        distana.set_weights(th_tools.load_model(params))

    """
    TRAINING
    """

    training_start_time = time.time()

     #
    # Start the training and iterate over all epochs
    for epoch in range(params['epochs']):
        epoch_start_time = time.time()

        # save batch errors
        training_errors = []

        # Iterate through epoch
        for _iter in range(amount_iter):

            # Train the network for the given training data
            mse = distana.train(iter_idx=_iter)

            training_errors.append(mse)
            
        supervisor.finished_training(training_errors)

        # Compute validation error
        # Evaluate and validate the network for the given validation data
        mse = distana.test(val_data_files)
        supervisor.finished_validation(mse)

        supervisor.finished_epoch(epoch, epoch_start_time)


    supervisor.finished(training_start_time)
    



if __name__ == "__main__":

    param_manager = DISTANAParams(FileManager(), description)
    
    th.manual_seed(42)

    # Load parameters from file
    params = param_manager.parse_params(is_training = True)

    print(f'''Run the training of architecture {
        params["architecture_name"]
        } with model {
        params["model_name"]
        } version {
        params["version_name"]
        } and data {
        params["data_type"]
        }''')
    run_training(params)








