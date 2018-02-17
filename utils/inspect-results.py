import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
from collections import OrderedDict

from rsub import *
from matplotlib import pyplot as plt

# --
# IO

history = {}
for f in sorted(glob('results/*')):
    try:
        history['.'.join(os.path.basename(f).split('.')[:-1])] = pickle.load(open(f, 'rb'))
    except:
        print('error at %s' % f)

# --
# Plot reward over time

for k,v in history.items():
    _ = plt.plot([vv['mean_reward'] for vv in v], label=k, alpha=0.75)

_ = plt.legend(loc='lower right', fontsize=6)
# _ = plt.ylim(0.8, 0.95)
show_plot()

# --
# Plot actions over time

action_counts = np.vstack([h['action_counts'] for h in history['pipenet-sgdr.0']])
for i, r in enumerate(action_counts.T):
    _ = plt.plot(r, alpha=0.5, label=i)

_ = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
_ = plt.ylim(0, 1)
show_plot()

z = (action_counts * 100).round().astype(int)
z[-50:]