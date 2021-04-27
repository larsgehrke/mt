import os
import numpy as np




batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
models = ["v1a", "v1b"]

for b in batches:
    for m in models:
        data = np.zeros(10)
        for i in range(10):
            x = os.popen(f"python test.py -m {m} -b {b}").read()
            data[i] = float(x)
            print(data[i])
        mean = np.mean(data)
        mean = np.around(mean, decimals=5)
        stddev = np.std(data)
        stddev = np.around(stddev, decimals = 5)
        print(f"[batch size {b}, model {m}] mean: {mean}, stddev: {stddev}")



