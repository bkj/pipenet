import os
import pickle
import numpy as np
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

# --
# IO

history = {}
for f in sorted(glob('results/*')):
    history[os.path.basename(f).split('.')[0]] = pickle.load(open(f, 'rb'))

# --
# Plot reward over time

for k,v in history.items():
    _ = plt.plot([vv['mean_reward'] for vv in v], label=k, alpha=0.75)

_ = plt.legend(loc='lower right')
_ = plt.ylim(0.8, 0.95)
show_plot()



# >>


df = pd.DataFrame(history['pipenet-qblock-2'])
(np.vstack(df.action_counts)[-50:] * 100).round().astype(int)
# <<

# --
# Plot actions over time

action_counts = np.vstack([h['action_counts'] for h in history['pipenet-qblock-2']])
for i, r in enumerate(action_counts.T):
    _ = plt.plot(r, alpha=0.5, label=i)

_ = plt.legend(loc='lower right')
_ = plt.ylim(0, 1)
show_plot()