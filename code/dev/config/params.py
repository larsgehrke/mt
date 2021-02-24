from pathlib import Path
import os
import argparse

class Params():

    def __init__(self, file_manager, folder):
        self.source_path = str(os.path.join(Path(), "config")) 
        self.file_manager = file_manager
        self.default_str = "default"
        self.folder = folder
        self.description = None
        self.init_values = None
        # organisational params, used for param management
        self.orga_params = ["params", "save_params", "reset_params", "verbose"]


    def _load(self, filename):

        if filename is None:
            return self._load_defaults()

        file_path = os.path.join(self.source_path, self.folder, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Tried to load the params but could not find %s" % file_path)

        else:
            return self.file_manager( mode = "load",
                name = file_path)

    def _save(self, filename, params_dict):
        file_path = os.path.join(self.source_path, self.folder, filename)
        
        self.file_manager( mode = "save",
                name = file_path,
                obj = params_dict)



    def _load_defaults(self):

        file_path = os.path.join(self.source_path , self.folder ,  self.default_str)
        
        if os.path.isfile(file_path + ".pkl"):
            return self.file_manager( mode = "load", 
                name = file_path)
        else:
            self.file_manager( mode = "save", 
                name = file_path,
                obj = self.init_values)
            return self.init_values

    def _replace_params(self, options_, params, mapping = None):
        options = vars(options_)

        if mapping is not None:
            for option_key,param_key in mapping.items():
                if options[option_key] is not None:
                    params[param_key] = options[option_key]

        else:
            for key, value in options.items():

                if value is not None and key not in self.orga_params:
                    params[key] = value

        return params

    def _reset_params(self):
        file_path = os.path.join(self.source_path , self.folder ,  self.default_str)
        self.file_manager( mode = "save", 
                name = file_path,
                obj = self.init_values)


    def _str2bool(self,v):
        '''
            Converting strings to boolean values
        '''
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


        