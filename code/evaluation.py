import os
import numpy as np




#batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
models = ["old", "old2"]
batches = ["","-g True"]
latex = ""

for b in batches:
    for i,m in enumerate(models):
        data = np.zeros(10)
        for i in range(10):
            command = f"python test.py {b} -m {m} -b 1"
            x = os.popen(command).read()
            data[i] = float(x)
        mean = np.mean(data)
        mean = np.around(mean, decimals=5)
        stddev = np.std(data)
        stddev = np.around(stddev, decimals = 5)
        print(f"[batch size {b}, model {m}] mean: {mean}, stddev: {stddev}")
        latex = latex + f"{mean} $\\pm$ {stddev}"
        if i < len(models)-1:
            latex = latex + " & "
    latex = latex + "\\\\\n"

print()
print(latex)



