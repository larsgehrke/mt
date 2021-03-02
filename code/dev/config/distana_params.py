'''

   This script holds the settings for the 
   training of DISTANA.

'''
import os
import argparse

from config.params import Params 
import tools.torch_tools as th_tools

class DISTANAParams(Params):

    def __init__(self, file_manager, description):

        super().__init__(file_manager, folder = "distana_params")

        self.description = description

        '''
            If you add/change something here, please add also some Argparse option further below
            and delete the current saved parameter files
        '''
        self.init_values = {

            "architecture_name": "distana",
            "model_name": "th",
            "version_name": "v1",
            "data_type": "tmp_data",

            "data_noise": 0.0, # 5e-5  # The noise that is added to the input data
            "p_zero_input": 0.0, # Probability of feeding no input to a PK

            "use_gpu": False, # for grid sizes > 25x25 the GPU version is faster
            "device": "cpu",

            #
            # Training parameters

            "save_model": True,
            "continue_training": False, 
            "epochs": 20,
            "seq_len": 140,
            "learning_rate": 0.001,
            "batch_size": 8,

            #
            # Testing parameters

            "teacher_forcing_steps": 15,
            "closed_loop_steps": 135,

            #
            # PK specific configurations    

            "pk_rows": 16, # Rows of PKs
            "pk_cols": 16, # Cols of PKs

            "pk_neighbors": 8,
            "pk_dyn_in_size": 1, # the z-value of the wave field
            "pk_lat_in_size": 1,
            "pk_pre_layer_size": 4,
            "pk_num_lstm_cells": 16,
            "pk_dyn_out_size": 1, # Must be equal to PK_DYN_IN_SIZE
            "pk_lat_out_size": 1  # Must be equal to PK_LAT_IN_SIZE       

        }

    def _save_paths(self, params):

        # Specify Paths for this program
        # Ending with directories
        params['data_folder'] = os.path.join(self.source_path, "data", params['data_type'], "")
        params['model_folder'] = os.path.join(self.source_path, "model", 
                                    params['model_name'], "saved_models", "") 

        return params

    def _save_additional_params(self, params):

        # Hide the GPU(s) in case the user specified to use the CPU in the config file
        if not params["use_gpu"]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Set device on GPU if specified in the params, else CPU
        params["device"] = th_tools.determine_device(params["use_gpu"])

        # Adding dependent parameters
        params['amount_pks'] = params['pk_rows'] * params['pk_cols']

        return params

    def parse_params(self, is_training):
        parser = argparse.ArgumentParser(description=self.description, 
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('--architecture-name', type=str, 
            help='The name of the architecture.')

        parser.add_argument('-m','--model-name', type=str, 
            help='The name of the model.')

        parser.add_argument('-v','--version-name', type=str, 
            help='The name of the model version.')

        parser.add_argument('-d','--data-type', type=str, 
            help='The type of the data. Used to load the data from this data subfolder.')

        parser.add_argument('--data-noise', type=float, 
            help='The noise that is added to the input data.')

        parser.add_argument('--p-zero-input', type=float, 
            help='Probability of feeding no input to a Prediction kernel.')

        parser.add_argument('-g', '--use-gpu', type=super()._str2bool, 
            help='''
            Should the GPU be used? Specify the type of device used for the computations. 
            For grid sizes > 25x25 the GPU version is faster
            ''')    

        parser.add_argument('--pk-rows', type=int, 
            help='Amount of PK (Prediction Kernel) rows.')      

        parser.add_argument('--pk-cols', type=int, 
            help='Amount of PK (Prediction Kernel) cols.')

        parser.add_argument('--pk-neighbors', type=int, 
            help='Amount of PK (Prediction Kernel) neighbors.')

        parser.add_argument('--pk-dyn-in-size', type=int, 
            help='The dynamical input size of one Prediction Kernel.')

        parser.add_argument('--pk-lat-in-size', type=int, 
            help='The lateral input size of one Prediction Kernel.')

        parser.add_argument('--pk-pre-layer-size', type=int, 
            help='The layer size of the first fully connected layer in the Prediction Kernel.')

        parser.add_argument('--pk-num-lstm-cells', type=int, 
            help='The number of LSTM cells.')

        parser.add_argument('--pk-dyn-out-size', type=int, 
            help='The dynamical output size of one Prediction Kernel.')

        parser.add_argument('--pk-lat-out-size', type=int, 
            help='The lateral output size of one Prediction Kernel.')


        # Saving and Loading parameter files
        parser.add_argument('-p','--params', type=str, 
            help='The file name of the parameter values to load.')

        parser.add_argument('--save-params', type=str, 
            help='''
            The file name where the parameter values should be stored. 
            With the name "default" the default values will be overwritten.
            ''')

        parser.add_argument('-r', '--reset-params', action='store_true', 
            help='Reset saved default parameter values to initial values.')

        parser.add_argument('--verbose', action='store_true', 
            help='Set the Verbosity of the program to True.')

        if is_training:
            parser = self._parse_train_params(parser)
        else:
            parser = self._parse_test_params(parser)


        options = parser.parse_args()

        # Constraints for the parser options
        # ...

        if options.reset_params:
            super()._reset_params()
        
        params = super()._load(options.params)

        params = super()._replace_params(options, params)

        params = self._save_additional_params(params)
        

        # Verbose
        if options.verbose:
            for p in params:
                print(f"{p}: {params[p]}")

        if options.save_params is not None:
            file = options.save_params
            super()._save(file, params)

        params = self._save_paths(params)




        return params


    def _parse_train_params(self, parser):

        # Options for training

        parser.add_argument('--save-model', type=super()._str2bool, 
            help='Whether the trained neural network model should be saved.')

        parser.add_argument('--continue-training', type=super()._str2bool,
            help='Whether the training of the neural network should be continued.')

        parser.add_argument('-e','--epochs', type=int, 
            help='The amount of epochs for the training.')

        parser.add_argument('-s','--seq-len', type=int, 
            help='The amount sequence length used for training.')     

        parser.add_argument('-l','--learning-rate', type=float, 
            help='Specify the learning rate for the training.')

        parser.add_argument('-b','--batch-size', type=int, 
            help='Specify the batch size for the training.')

        return parser

        


    def _parse_test_params(self, parser):

        # Options for testing

        parser.add_argument('--teacher-forcing-steps', type=int, 
            help='Amount of time steps for the teacher forcing.')

        parser.add_argument('--closed-loop-steps', type=int, 
            help='Amount of time steps for the closed loop.')

        return parser




    
