'''

   This script is responsible for reading and 
   writing Python objects into files. 

'''

#
# The glob module finds all the pathnames matching a specified pattern 
# according to the rules used by the Unix shell, 
# although results are returned in arbitrary order.
import glob
import numpy as np

def get_data_filenames(path_to_dir: str) -> list:
    '''
        Collect the data file names of a given path.
    '''

    #
    # Get the training and validation file names
    data_filenames = np.sort(glob.glob(path_to_dir + '*'))

    if len(data_filenames) == 0:
        raise Exception('Could not find training data in '+ str(path_to_dir) + '*')

    return data_filenames

import pickle
import os

class FileManager():
    '''
        Saves and Loads Python objects to/from files. 
        For this the python library pickle is used, that is why the suffix is 'pkl.'.
    '''

    def __init__(self):
        self.suffix = '.pkl'

    def save(self, path: str, name: str, obj: dict):
        '''
        Saves a python dictionary to file.
        '''
        file = name + self.suffix

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path,file), 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name: str) -> dict:
        '''
        Loads a python dictionary from file.
        '''
        with open('' + name + self.suffix, 'rb+') as f:
            return pickle.load(f)

    def isfile(self, name: str) -> bool:
        '''
        Checks if a certain file exists.
        '''
        return os.path.isfile(name+self.suffix)

    def __call__(self, mode: str, path: str, name: str, obj:dict = None):
        '''
        This class can be called with three different modes: save, load or load_or_save.
        - save: Saves the given object to file
        - load: Loads a python objects specified by the given path + name
        - load_or_save: tries to load if file exists, saves the object otherwise 
        '''
        modes = ["save", "load", "load_or_save"]

        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)

        elif mode == "save":
            if obj is None:
                raise ValueError("The object to save as file is None.")

            self.save(path, name, obj)

        elif mode == "load":
            file_path = os.path.join(path,name)
            return self.load(file_path)

        elif mode == "load_or_save":
            file_path = os.path.join(path,name) 
            if self.isfile(file_path):
                return self.load(name = file_path)
            else:
                if obj is None:
                    raise ValueError("The object to save as file is None.")
                self.save( path = path,
                    name = name,
                    obj = obj)
                return obj


