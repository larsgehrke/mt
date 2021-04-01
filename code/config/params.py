from pathlib import Path
import os
import argparse

from tools.persistence import FileManager

class Params():
    '''
    This is the superclass for all Parameter subclasses, responsible for managing the parameters, 
    e.g. commission the loading and saving of parameters to file.
    '''

    def __init__(self, folder: str):
        '''
        Initialisation of the Params superclass.
        :param file_manager: object that is doing the file persistence
        :param folder: name of use case, in this folder the config files are saved
        '''
        # Import Paths
        self.source_path = str(Path()) 
        self.config_path = str(os.path.join(self.source_path, "config")) 

        # Creating object that is responsible for the persistence layer
        self.file_manager = FileManager()

        # saving argument to class scope
        self.folder = folder

        # string to identify the default config file
        self.default_str = "default"
        
        # description of use case and 
        self.description = None # specified by subclasses

        # first default parameters if no default config file exists
        self.init_values = None # specified by subclasses

        # these cl arguments are ignored, 
        # because the subclasses use them for management purposes
        self.orga_params = ["params", "save_params", "reset_params", "verbose"]

    @staticmethod
    def to_string(dict_obj: dict) -> str:
        '''
        Getting the string represantation of the parameter dictionary.
        :param dict_obj: parameters as dictonary
        '''
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



    def _load(self, file_name: str) -> dict:
        '''
        Load the parameters from file.
        :param file_name: config file name
        '''

        if file_name is None:
            return self._load_defaults()
        else:
            return self.file_manager( mode = "load",
                path = self._get_path(),
                name = file_name)

    def _save(self, file_name: str, params_dict: dict):
        '''
        Save the parameters to file.
        :param file_name: file name of the parameters to be saved
        :param params_dict: parameters as dictionary
        '''        
        
        self.file_manager( mode = "save",
                path = self._get_path(),
                name = file_name,
                obj = params_dict)



    def _load_defaults(self) -> dict:
        '''
        Either load the default file if it exists 
        or save the init_values as the new default file.
        In both cases return the corresponding parameter dict.
        This is outsourced to the file manager object.
        '''

        return self.file_manager( mode = "load_or_save", 
                path = self._get_path(),
                name = self.default_str,
                obj = self.init_values)

    def _replace_params(self, options_: dict, params: dict, mapping:dict = None) -> dict:
        '''
        This method is overwriting specific parameters 
        that are set by command line arguments.
        Every cl argument has the default value None. This method only overwrites
        parameter values, if the cl argument is not None. Thus only cl arguments,
        that were really set by the user are taken into account.
        :param options_: command line values
        :param params: loaded parameters
        :param mapping: can be specified if the keys of options_ and params are not the same 
        '''
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
        '''
        Commission the file manager to save the init_values as the new default parameter file.
        Thus overwrite the old default parameter file.
        '''
        self.file_manager( mode = "save", 
                path = self._get_path(),
                name = self.default_str,
                obj = self.init_values)

    def _get_path(self) -> str:
        '''
        Get the path to the folder in which the parameter files 
        for the specific use case are saved.
        '''
        return os.path.join(self.config_path , self.folder )


    def _str2bool(self,v) -> bool:
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


        