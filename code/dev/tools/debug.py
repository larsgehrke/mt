import numpy as np
import torch as th
import sys
import time


def sprint(obj, obj_name="Object", complete=False, exit=False):
    print("Printing out", obj_name)
    print(type(obj))

    if (isinstance(obj, th.Tensor)):
        obj = obj.cpu().detach().numpy()

    if (isinstance(obj, np.ndarray)):
        print(obj.shape)

    if (complete):
        print(obj)

    if(exit):
        sys.exit()

def lprint(l, exit=False):
    for i in l:
        print(i)

    if exit:
        sys.exit()

class Clock():
    def __init__(self, s):
        print("Starting the clock at "+ str(s))
        self.total = time.time()
        self.t = time.time()

    def split(self, s=""):
        print(str(s)+ ": ", str(np.round(time.time() - self.t, 6)).ljust(6, '0') + " seconds")
        self.t = time.time()

    def stop(self, exit = True):
        print("Total time: " + str(np.round(time.time() - self.total, 6)).ljust(6, '0') + " seconds")
        if exit:
            sys.exit()
