#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


fast_file_name = 'photobleaching_mixture00_grid.csv'
slow_file_name = 'photobleaching_mixture01_grid.csv'

data_fast = np.genfromtxt(fast_file_name, delimiter = ',', skip_header = True)
data_slow = np.genfromtxt(slow_file_name, delimiter = ',', skip_header = True)

# features are both the same in each case
features     = data_fast[:, :-1]
fast_targets = data_fast[:, -1]
slow_targets = data_fast[:, -1]

data = {'features': features, 'fast_targets': fast_targets, 'slow_targets': slow_targets }
print(data)
print(data['features'].shape, data['fast_targets'].shape, data['slow_targets'].shape)

with open('dataset.pkl', 'wb') as content:
	pickle.dump(data, content)
