import os
import numpy as np




batches = ["1", "10", "100"]
models =  ["old", "old2", "v1a", "v1b", "v2", "v3"]
settings = [""]

desc_eval = ""
desc_eval_file = "desc_eval.txt"
plain_eval =  ""
plain_eval_file = "plain_eval.txt"

counter = 0

for b in batches:
    for m in models:
        for setting in settings:
            #data = np.zeros(10)
            desc = f"\n\n{counter}) batch size: {b} model: {m} settings: {setting}\n"
            counter += 1
            print(desc)
            desc_eval += desc
            command = f"python train.py -b {b} -m {m} {setting}"
            x = os.popen(command).read()
            print(x)
            plain_eval+=x
            desc_eval+=x
            plain_eval += "===\n"

    #   lines = x.split("\n")
    #     for line in lines:
    #         speed, train, val = line.split(",")
    #         data[i] = float(speed)
    #     mean = np.mean(data)
    #     mean = np.around(mean, decimals=5)
    #     stddev = np.std(data)
    #     stddev = np.around(stddev, decimals = 5)
    #     print(f"[batch size {b}, model {m}] mean: {mean}, stddev: {stddev}")
    #     latex = latex + f"{mean} $\\pm$ {stddev}"
    #     if idx < len(models)-1:
    #         latex = latex + " & "
    # latex = latex + "\\\\\n"

with open(desc_eval_file, "w+") as f:
    f.write(desc_eval)

with open(plain_eval_file, "w+") as f:
    f.write(plain_eval)




