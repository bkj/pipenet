from data import *
from time import time
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

loaders = make_cifar_dataloaders(num_workers=8, pin_memory=False)

times = []
for epoch in range(3):
    t = time()
    tmp_times = []
    for x,y in loaders['train']:
        tmp_times.append(time() - t)
    times.append(tmp_times)
    print(epoch)

print([t[-1] for t in times])
print(sum([t[-1] for t in times]))