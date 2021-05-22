import os
import numpy as np




batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
old_models = ["old", "old2"]
new_models =  ["v1a", "v1b", "v2", "v3"]
device = ["False", "True"]

rows = []
row = ""
for d in device:
    print(f"=== use gpu? {d} ===")
    for om in old_models:
        print(f"= version {om} =")
        command = f"python train.py --use-gpu {d} -b 1 -m {om}"
        x = os.popen(command).read()
        print(x)
    print()
    for b in batches:
        print("== batch size " + str(b)+ " ==")
        for nm in new_models:
            print(f"= version {nm} =")
            command = f"python train.py --use-gpu {d} -b {b} -m {nm}"
            x = os.popen(command).read()
            print(x)
        print()

