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
        self.times = []
        self.counter = 0

    def split(self, s=""):
        diff = time.time() - self.t
        print(str(s)+ ": ", str(np.round(diff, 6)).ljust(6, '0') + " seconds")
        self.t = time.time()
        return diff

    def split_means(self):
        diff = time.time() - self.t
        self.counter += 1
        self.times.append(diff)
        if self.counter % 1000 == 0:
            print(np.mean(self.times))
        self.t = time.time()

    def reset(self):
        self.t = time.time()

    def stop(self, exit = True):
        print("Total time: " + str(np.round(time.time() - self.total, 6)).ljust(6, '0') + " seconds")
        if exit:
            sys.exit()
