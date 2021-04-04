#!/usr/bin/env python

import numpy as np

from gemini import Gemini
#===============================================================================

'''
Train Gemini on 1d regression example
'''

#===============================================================================

# define the 1d surfaces
def fast_func(x):
    return  np.sin(4 * np.pi * x)

def slow_func(x):
    return np.cos(3 * np.pi * x)+0.5



# define the parameters
NUM_SLOW_TRAIN = 4
NUM_FAST_TRAIN = 50
NUM_SLOW_VALID = 100
NUM_FAST_VALID = 100

# generate the data
np.random.seed(100697)
train_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_TRAIN, 1))
train_slow_targets  = slow_func(train_slow_features)

train_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_TRAIN, 1))
train_fast_targets  = fast_func(train_fast_features)

valid_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_VALID, 1))
valid_slow_targets  = slow_func(valid_slow_features)

valid_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_VALID, 1))
valid_fast_targets  = fast_func(valid_fast_features)


# intantiate gemini
gemini = Gemini()
# train gemini with plotting
gemini.train(train_slow_features, train_slow_targets,
             train_fast_features, train_fast_targets,
             valid_slow_features, valid_slow_targets,
             valid_fast_features, valid_fast_targets,
             num_folds=1,
             user_hyperparams={'max_epochs': 30000,
                               'es_patience': None,},
             plot=True)
