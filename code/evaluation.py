import os
import numpy as np




batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
models =  ["v1a", "v1b", "v2", "v3"]

rows = []
row = ""

for idx,b in enumerate(batches):
    row[idx] = str(b)
    print("batch size " + str(b)+ ":")
    for m in models:
        res = "model " + str(m) + ": "
        command = f"python train.py -b {b} -m {m}"
        x = os.popen(command).read()
        print(res + str(x))
        row[idx] = row[idx] + " & " + str(x)  
        
    row[idx] = row[idx] + "\\\\"

for r in rows:
    print(str(r))

