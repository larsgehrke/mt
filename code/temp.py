import numpy as np

with open('cpu.npy', 'rb') as f:
    cpu = np.load(f)

with open('gpu.npy', 'rb') as f:
    gpu = np.load(f)


print(np.array_equal(cpu, gpu))

print(cpu)

print(gpu)