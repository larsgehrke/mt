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

from model.distana import DISTANA
from config.distana_params import DISTANAParams
from config.params import Params
from tools.persistence import FileManager
from tools.persistence import get_data_filenames
from tools.supervisor import TestSupervisor
import tools.torch_tools as th_tools




def run_testing(params):

    # Create and set up the network
    distana = DISTANA(params)

    criterion = th.nn.MSELoss() 

    # Load the trained weights from file
    distana.set_weights(th_tools.load_model(params), is_training=False)
    
    # Get the test data
    test_data_files = get_data_filenames(os.path.join(params['data_folder'],'test',''))
    # Save the test configuration and data in the model class
    amount_test = distana.set_testing(test_data_files,criterion, params['teacher_forcing_steps'])
    # Create the view object 
    supervisor = TestSupervisor(params,distana.get_trainable_params())

    

    x = u'1'
    batch_idx, sample_idx = 0, 0
    mse, net_outputs_batch, net_label_batch, net_input_batch = None, None, None, None

    while x == u'1' and batch_idx<amount_test:

        if sample_idx == 0:
            time_start = time.time()

            # Evaluate the network for the given test data
            mse, net_outputs, net_label, net_input = distana.test(iter_idx=batch_idx)
            
            forward_pass_duration = time.time() - time_start
            print("\tForward pass for batch size ",params["batch_size"]," took: ", forward_pass_duration, " seconds.")

        if params["batch_size"] == 1:
            # when the model cannot handle batches
            supervisor.plot_sample(net_outputs, 
                net_label, net_input)
        else:
            # when the model is able to handle batches
            supervisor.plot_sample(net_outputs[sample_idx], 
                net_label[sample_idx], net_input[sample_idx])

        # Retrieve user input to continue or quit the testing
        x = input("Press 1 to see another example, anything else to quit.")
        
        sample_idx += 1
        
        if sample_idx >= params["batch_size"]:
            batch_idx += 1 
            sample_idx = 0 

    supervisor.finished()




if __name__ == "__main__":

    param_manager = DISTANAParams(FileManager(), description)
    
    th.manual_seed(42)

    # Load parameters from file
    params = param_manager.parse_params(is_training = False)

    print(f'''Run the testing of architecture {
        params["architecture_name"]
        } with model {
        params["model_name"]
        } and data {
        params["data_type"]
        }''')
    run_testing(params)








