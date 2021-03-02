from pathlib import Path
import os
import argparse

class Params():

    def __init__(self, file_manager, folder):
        self.source_path = str(Path()) 
        self.config_path = str(os.path.join(self.source_path, "config")) 
        self.file_manager = file_manager
        self.default_str = "default"
        self.folder = folder
        self.description = None # specified by child classes
        self.init_values = None # specified by child classes
        # organisational params, used for param management
        self.orga_params = ["params", "save_params", "reset_params", "verbose"]

    @staticmethod
    def to_string(dict_obj):
        res = "{" + os.linesep
        for key, value in dict_obj.items():
            res += '"' + str(key) + '": '
            if type(value) == str:
                res += '"' + str(value) + '"'
            else:
                res += str(value)
            res += ","+os.linesep
        res += "}"

        return res



    def _load(self, file_name):

        if file_name is None:
            return self._load_defaults()
        else:
            return self.file_manager( mode = "load",
                path = self._get_path(),
                name = file_name)

    def _save(self, file_name, params_dict):
        
        self.file_manager( mode = "save",
                path = self._get_path(),
                name = file_name,
                obj = params_dict)



    def _load_defaults(self):

        return self.file_manager( mode = "load_or_save", 
                path = self._get_path(),
                name = self.default_str,
                obj = self.init_values)

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
        self.file_manager( mode = "save", 
                path = self._get_path(),
                name = self.default_str,
                obj = self.init_values)

    def _get_path(self):
        return os.path.join(self.config_path , self.folder )


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


        