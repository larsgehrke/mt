import os
import numpy as np




batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
models =  ["v1a", "v1b", "v2", "v3"]

out = ""

for b in batches:
    for m in models:
        command = f"python train.py -b {b} -m {m}"
        x = os.popen(command).read()
        out = out + " & " + str(x)  
        
    print(b + out +"\\\\")


print()
print()
print()
out = "Order batches: "
for b in batches:
    out = out + str(b)+", "
print(out)
out = "Order models: "
for m in modelss:
    out = out + str(m)+", "






