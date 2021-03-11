'''

   This script is responsible for reading and 
   writing Python objects into files. 

'''
# The glob module finds all the pathnames matching a specified pattern 
# according to the rules used by the Unix shell, 
# although results are returned in arbitrary order.
import glob
import numpy as np


def get_data_filenames(path_to_dir):
    #
    # Get the training and validation file names
    data_filenames = np.sort(glob.glob(path_to_dir + '*'))

    if len(data_filenames) == 0:
        raise Exception('Could not find training data in '+ str(path_to_dir) + '*')

    return data_filenames

import pickle
import os

class FileManager():

    def __init__(self):
        self.suffix = '.pkl'

    def save(self,path, name, obj):
        file = name + self.suffix

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path,file), 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self,name):
        with open('' + name + self.suffix, 'rb+') as f:
            return pickle.load(f)

    def isfile(self, name):
        return os.path.isfile(name+self.suffix)

    def __call__(self, mode, path, name, obj=None):
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


