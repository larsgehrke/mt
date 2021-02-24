'''

   This script is responsible for reading and 
   writing Python objects into files. 

'''

import pickle
import os

class FileManager():

    def save(self,obj, name ):
        if not os.path.exists(name):
            os.makedirs(name)

        with open(name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self,name):
        with open('' + name + '.pkl', 'rb+') as f:
            return pickle.load(f)

    def __call__(self, mode, name, obj=None):
        modes = ["save", "load"]

        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)

        elif mode == "save":
            self.save(obj,name)

        elif mode == "load":
            return self.load(name)