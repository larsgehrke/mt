import os
import numpy as np




batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
old_models = ["old", "old2"]
new_models =  ["v1a", "v1b", "v2", "v3"]
use_gpu_values = ["False", "True"]

rows = []
row = ""
for use_gpu in enumerate(use_gpu_values):
    print(f"=== use gpu? {use_gpu} ===")
    for om in old_models:
        print(f"= version {om} =")
        command = f"python train.py --use-gpu {use_gpu} -b 1 -m {om}"
        x = os.popen(command).read()
        print(x)
    print()
    for b in batches:
        print("== batch size " + str(b)+ " ==")
        for nm in enumerate(new_models):
            if not(use_gpu=="False" and (nm=="v2" or nm=="v3")):   
                print(f"= version {nm} =")
                command = f"python train.py --use-gpu {use_gpu} -b {b} -m {nm}"
                x = os.popen(command).read()
                print(x)
        print()

