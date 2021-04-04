#!/usr/bin/env python

import pytest
import numpy as np

from gemini import Gemini
from gemini import GeminiOpt
#======================================================================


def fast_func(x):
    return  np.sin(4 * np.pi * x)

def slow_func(x):
    return np.cos(3 * np.pi * x)

def test_gemini():
    '''
    Test training of BNN on analytic functions by interacting with BNN
    from the model level (lower level implementation)
    '''
    NUM_SLOW_TRAIN = 6
    NUM_FAST_TRAIN = 30
    NUM_SLOW_VALID = 100
    NUM_FAST_VALID = 100

    np.random.seed(100694)
    train_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_TRAIN, 1))
    train_slow_targets  = slow_func(train_slow_features)

    train_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_TRAIN, 1))
    train_fast_targets  = fast_func(train_fast_features)

    valid_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_VALID, 1))
    valid_slow_targets  = slow_func(valid_slow_features)

    valid_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_VALID, 1))
    valid_fast_targets  = fast_func(valid_fast_features)

    gemini = Gemini()
    hyperparams = {'max_epochs': 1000} # short training for ci
    gemini.train(train_slow_features, train_slow_targets,
                 train_fast_features, train_fast_targets,
                 valid_slow_features, valid_slow_targets,
                 valid_fast_features, valid_fast_targets,
                 num_folds=2, user_hyperparams=hyperparams)



def test_gemini_opt():

    NUM_SLOW_TRAIN = 6
    NUM_FAST_TRAIN = 30
    NUM_SLOW_VALID = 100
    NUM_FAST_VALID = 100

    np.random.seed(100694)
    train_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_TRAIN, 1))
    train_slow_targets  = slow_func(train_slow_features)

    train_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_TRAIN, 1))
    train_fast_targets  = fast_func(train_fast_features)

    valid_slow_features = np.random.uniform(low=0, high=1, size=(NUM_SLOW_VALID, 1))
    valid_slow_targets  = slow_func(valid_slow_features)

    valid_fast_features = np.random.uniform(low=0, high=1, size=(NUM_FAST_VALID, 1))
    valid_fast_targets  = fast_func(valid_fast_features)



    gemini = GeminiOpt()
    hyperparams = {'max_epochs': 1000} # short training for ci
    gemini.train(train_slow_features, train_slow_targets,
                 train_fast_features, train_fast_targets,
                 valid_slow_features, valid_slow_targets,
                 valid_fast_features, valid_fast_targets,
                 num_folds=2, user_hyperparams=hyperparams)

    pearson = gemini.get_pearson_coeff()

    assert(type(pearson) == np.float64)
    assert(-1. <= pearson <= 1.)


if __name__ == '__main__':
    test_gemini()
    test_gemini_opt()
