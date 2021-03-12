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

from model.facade import Facade
from config.distana_params import DISTANAParams
from config.params import Params
from tools.persistence import FileManager
from tools.persistence import get_data_filenames
from tools.supervisor import TrainSupervisor
import tools.torch_tools as th_tools

from tools.debug import Clock



def run_training(params):

    # Create and set up the network
    model = Facade(params)
    
    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.net().parameters(), lr=params['learning_rate'])
    criterion = th.nn.MSELoss() 

    # Get the training and validation data
    train_data_files = get_data_filenames(os.path.join(params['data_folder'],'train',''))
    val_data_files = get_data_filenames(os.path.join(params['data_folder'],'val',''))

    # Set the training parameters for the model 
    # and get the amount of iterations for one epoch
    amount_train = model.set_training(train_data_files, optimizer, criterion)
    amount_val = model.set_testing(val_data_files, criterion, params['teacher_forcing_steps'])

    model_saver = None
    if params["save_model"]:
        model_saver = th_tools.Saver(epochs = params["epochs"],
            model_src_path=params['model_folder'],
            model_name = params['model_name'],
            cfg_file=Params.to_string(params), 
            net=model.net())
    
    view = TrainSupervisor(params['epochs'], model.get_trainable_params(), model_saver)

    if params["continue_training"]:
        model.set_weights(th_tools.load_model(params),is_training=True)

    """
    TRAINING
    """

    training_start_time = time.time()

     #
    # Start the training and iterate over all epochs
    for epoch in range(params['epochs']):
        c = Clock(" === Starting Epoch " + str(epoch))
        epoch_start_time = time.time()

        # save batch errors
        training_errors = []
        val_errors = []

        # Iterate through epoch
        for _iter_train in range(amount_train):
            # Train the network for the given training data
            mse = model.train(iter_idx=_iter_train)
            c.split_means(f"Epoch {epoch} training iteration {_iter_train}")

            training_errors.append(mse)

        print(f"Epoch {epoch} Iteration ended")
        view.finished_training(training_errors)

        # Iterate through epoch
        for _iter_val in range(amount_val):
            # Test the network for the given validation data
            mse = model.test(iter_idx=_iter_val)
            c.split(f"Epoch {epoch} validation iteration {_iter_val}")

            val_errors.append(mse)
        print(f"Epoch {epoch} Validation ended")
        view.finished_validation(val_errors)
        

        view.finished_epoch(epoch, epoch_start_time)
        sprint(f"Epoch {epoch} ended")


    view.finished(training_start_time)
    



if __name__ == "__main__":

    param_manager = DISTANAParams(FileManager(), description)
    
    th.manual_seed(42)

    # Load parameters from file
    params = param_manager.parse_params(is_training = True)

    print(f'''Run the training of architecture {
        params["architecture_name"]
        } with model {
        params["model_name"]
        } and data {
        params["data_type"]
        }''')
    run_training(params)








