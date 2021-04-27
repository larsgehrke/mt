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
import os
import time
import numpy as np
import torch as th 

from model.facade import Facade
from config.distana_params import DISTANAParams
from config.params import Params
from tools.persistence import FileManager
from tools.persistence import get_data_filenames
from tools.supervisor import TestSupervisor
import tools.torch_tools as th_tools


def run_testing(params):

    # Create and set up the network
    model = Facade(params)

    # Set up the criterion (loss)
    criterion = th.nn.MSELoss() 

    # Load the trained weights from file
    model.set_weights(th_tools.load_model(params), is_training=False)
    
    # Get the test data
    test_data_files = get_data_filenames(os.path.join(params['data_folder'],'test',''))
    # Save the test configuration and data in the model class and get max number of iterations
    amount_test = model.set_testing(test_data_files,criterion, params['teacher_forcing_steps'])
    # Create the view object 
    view = TestSupervisor(params,model.get_trainable_params())

    
    # save unicode char 1 for the loop later
    x = u'1'

    # get the current batch size
    batch_size = model.config().batch_size_test

    #
    # creating variables for the iterations of the test loop 
    batch_idx, sample_idx = 0, 0
    error, net_outputs_batch, net_label_batch, net_input_batch = None, None, None, None

    # if 1 was pressed and batch index is less than max number of iterations
    while x == u'1' and batch_idx<amount_test: 

        if sample_idx == 0: # just do the forward pass once per batch
            time_start = time.time() 

            # Evaluate the network for the given test data
            error, net_outputs, net_label, net_input = model.test(iter_idx=batch_idx, 
                return_only_error = False)
            
            view.finish_batch(time_start, batch_size, error)

        # Plot or save sample if desired
        view.plot_sample(net_outputs[sample_idx], 
            net_label[sample_idx], net_input[sample_idx])

        # Retrieve user input to continue or quit the testing
        x = 2 #input("Press 1 to see another example, anything else to quit.")
        
        sample_idx += 1
        
        #
        # if current batch is finished, start new batch
        if sample_idx >= batch_size:
            batch_idx += 1 
            sample_idx = 0 

    # tell view that program is finished
    view.finished()



if __name__ == "__main__":
    '''Starting point of program'''

    # Get the parameter handler for the model
    param_manager = DISTANAParams(description)
    
    # set PyTorch seed for reproducibility
    th.manual_seed(42)

    # Load test parameters from file
    params = param_manager.parse_params(is_training = False)

    # print out basic information about this run
    print(f'''Run the testing of architecture {
        params["architecture_name"]
        } with model {
        params["model_name"]
        } and data {
        params["data_type"]
        }''')

    # call main method above
    run_testing(params)








