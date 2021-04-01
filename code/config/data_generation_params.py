
import os
from config.params import Params 
import argparse

class DataGenerationParams(Params):
    '''
    Specifying the parameters and command line arguments 
    for the use case of the data generation.
    The superclass Params is doing the management of the parameters.
    '''

    def __init__(self, description: str):
        '''
        Initialisation of DataGenerationParams.
        :param description: descripton of use case.
        '''

        super().__init__("data_generation_params")

        # saving argument to (super) class scope
        self.description = description
        
        #
        # defining the initial parameter values for the data generation
        # note that these values will only be used for the very first execution.
        # After that, the default file will always be loaded if it exists
        # or a user specific file, if it was specified by cl arguments
        # 
        # if you change keys or add keys here, 
        # please delete all created config files (.pkl) 
        # in config/data_generation_params/ on all devices 
        # and update cl arguments below
        self.init_values = {

        "threshold_empty_data": 0.00,  # In 1%, an empty data set (only zeros) is created

        "data_name": os.path.join("data", str('tmp_data')), 

        "save_data": False,
        "visualize": False,

        #
        # Field parameters
        "width": 16,  # Width of the simulated field in pixels
        "height": 16,  # Height of the simulated field in pixels


        # Saving parameters
        "create_n_files": 1,  # The number of data files that shall be created
        "data_set": 'train',  # Can be 'test', 'train' or 'val'

        "time_steps": 150,  # Number of simulation steps

        # Simulation parameters
        "dt": 0.1,  # Temporal step size
        "dx": 1,  # Step size in x-direction
        "dy": 1,  # Step size in y-direction

        #
        # Wave parameters
        "wave_width_x": 0.5,  # Width of the wave in x-direction
        "wave_width_y": 0.5,  # Width of the wave in y-direction
        "amplitude": 0.34,  # Amplitude of the wave
        "velocity": 3.0,  # The velocity of the wave
        "waves": 1,  # The number of waves that propagate simultaneously in one sequence
        "damp": 1.0,  # How much a wave is dampened (decaying over time)

        "variable_velocity": False,  # If true, waves of different speed are generated
        "variable_amplitude": False,  # If true, waves have different amplitudes
        "velocities": [0.5, 5.0],  # The min and max values for the wave speed
        "amplitudes": [0.1, 0.34]  # The min and max values for the wave amplitude

        }

    def parse_params(self) -> dict:
        ''' 
        Parsing Options from the command line
        :return: parameters as dictionary
        '''

        parser = argparse.ArgumentParser(description=self.description, 
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('mode', choices=['show', 'save', 'show_save'], 
            help='''
            Shall the generated data be visualized, saved to file or both?
            ''')

        parser.add_argument('-n', '--name', type=str, 
            help='''
            The name of the parent folder of the generated data.
            ''')

        parser.add_argument('-d','--data-set', choices=['train', 'val', 'test'], 
            help='''
            The name of the subfolder of the generated data.
            ''')

        parser.add_argument('-f', '--files', type=int, 
            help='''
            The number of data files that shall be created.
            ''')

        parser.add_argument('-t', '--time-steps', type=int, 
            help='''
            The number of simulation steps.
            ''')

        parser.add_argument('--width', type=int, 
            help='''
            Width of the simulated field in pixels.
            ''')
        parser.add_argument('--height', type=int, 
            help='''
            Height of the simulated field in pixels.
            ''')


        parser.add_argument('-p','--params', type=str, 
            help='''
            The file name of the parameter values to load.
            ''')

        parser.add_argument('--save-params', type=str, 
            help='''
            The file name where the parameter values should be stored.
            With the name "default" the default values will be overwritten.
            ''')

        parser.add_argument('-r', '--reset-params', action='store_true', 
            help='''
            Reset saved default parameter values to initial values.
            ''')

        options = parser.parse_args()

        # Constraints for the parser options
        if options.mode in ['save', 'show_save']:
            if options.data_set is None:
                parser.error('''
                    You must define the subfolder (-d {train, val, test'}),
                    where the generated data will be stored.
                    ''')
        
        if options.reset_params:
            super()._reset_params() 
        
        params = super()._load(options.params)

        #
        # Mapping from options key to params key
        mapping= {

        "data_set": "data_set",
        "files": "create_n_files",
        "time_steps": "time_steps",
        "width": "width",
        "height": "height"
        }

        params["save_data"] = options.mode in ['save', 'show_save']
        params["visualize"] = options.mode in ['show', 'show_save']

        if options.name is not None:
            params["data_name"] = os.path.join("data", options.name)

        params = super()._replace_params(options, params, mapping)

        # Verbose
        print(f"Amount of files: {params['create_n_files']}")
        print(f"Timesteps: {params['time_steps']}")

        if options.save_params is not None:
            file = options.save_params
            super()._save(file, params)

        return params



    
