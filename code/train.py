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

import time
import sys



def run_training(params):
    '''Running the training from top-level perspective, calling sub modules'''

    # Create and set up the network
    model = Facade(params)
    
    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.net().parameters(), lr=params['learning_rate'])
    criterion = th.nn.MSELoss()
    

    # Get the training and validation data
    train_data_files = get_data_filenames(os.path.join(params['data_folder'],'train',''))
    val_data_files = get_data_filenames(os.path.join(params['data_folder'],'val',''))

    # Set the training and validation parameters for the model 
    # and get the amount of iterations for one epoch respectively
    amount_train = model.set_training(train_data_files, optimizer, criterion)
    amount_val = model.set_testing(val_data_files, criterion, params['teacher_forcing_steps'])

    model_saver = None
    if params["save_model"]: # if the model should be saved
        # configure Saver object
        model_saver = th_tools.Saver(epochs = params["epochs"],
            model_src_path=params['model_folder'],
            model_name = params['model_name'],
            cfg_file=Params.to_string(params), 
            net=model.net())
    
    # instantiate a view object to process training results (print to console and save model)
    view = TrainSupervisor(params['epochs'], model.get_trainable_params(), model_saver)

    if params["continue_training"]: # load model if desired
        model.set_weights(th_tools.load_model(params),is_training=True)

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
        val_errors = []
        time_train = []

        # Iterate through epoch
        for _iter_train in range(amount_train)[:10]:
            # Train the network for the given training data
            before = time.time()
            mse = model.train(iter_idx=_iter_train)
            dur = time.time()-before

            print(f"{amount_train + 1 } iteration: {str(np.round(dur, 3))} seconds")
            time_train.append(dur)

            # collect training errors
            training_errors.append(mse)
            
        mean = np.mean(time_train)
        mean = np.around(mean, decimals=5)
        stddev = np.std(time_train)
        stddev = np.around(stddev, decimals = 5)
        print(f"mean: {mean}, stddev: {stddev}")

        sys.exit()

        # process training results
        view.finished_training(training_errors)
        # Retrieve user input to continue or quit the testing
        x = input("Press 1 to see another example, anything else to quit.")

        # Iterate through epoch
        for _iter_val in range(amount_val):
            # Test the network for the given validation data
            mse = model.test(iter_idx=_iter_val)

            # collect validation errors
            val_errors.append(mse)
            
        # process validation results
        view.finished_validation(val_errors)
        
        # print out epoch results
        view.finished_epoch(epoch, epoch_start_time)

    # print out end of training
    view.finished(training_start_time)
    


if __name__ == "__main__":
    '''Starting point of program'''

    # Get the parameter handler for the model
    param_manager = DISTANAParams(description)
    
    # set PyTorch seed for reproducibility
    th.manual_seed(42)

    # Load train parameters from file
    params = param_manager.parse_params(is_training = True)

    # print out basic information about this run
    # print(f'''Run the training of architecture {
    #     params["architecture_name"]
    #     } with model {
    #     params["model_name"]
    #     } and data {
    #     params["data_type"]
    #     }''')

    # call main method above
    run_training(params)








