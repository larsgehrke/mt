import os
import numpy as np

x = os.popen("python test.py -m v1a -b 64").read()
print(x)
