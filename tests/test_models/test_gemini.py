#!/usr/bin/env python

import pytest
import numpy as np

from gemini.data_transformer import DataTransformer
from gemini.models.gemini import Gemini

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
    NUM_SLOW_TRAIN = 4
    NUM_FAST_TRAIN = 20
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
    gemini.max_epochs = 1*10**3 # short training time for tests
    gemini.build_graph()
    gemini.build_inference()
    train_results, valid_results, variables = gemini.train(train_fast_features, train_fast_targets,
                                                           valid_fast_features, valid_fast_targets,
                                                           train_slow_features, train_slow_targets,
                                                           valid_slow_features, valid_slow_targets,
                                                           model_path=None, compute_loss=True)
    assert(type(train_results) == dict)
    assert(type(valid_results) == dict)

    features = np.random.uniform(0, 1, (10, 1))

    mean_pred_slow, std_pred_slow = gemini.run_prediction(features, 'slow', scaled=False)

    assert(type(mean_pred_slow) == np.ndarray)
    assert(type(std_pred_slow) == np.ndarray)
    assert(mean_pred_slow.shape == features.shape)
    assert(std_pred_slow.shape == features.shape)

    mean_pred_fast, std_pred_fast = gemini.run_prediction(features, 'fast', scaled=False)

    assert(type(mean_pred_fast) == np.ndarray)
    assert(type(std_pred_fast) == np.ndarray)
    assert(mean_pred_fast.shape == features.shape)
    assert(std_pred_fast.shape == features.shape)



if __name__ == '__main__':
    test_gemini()
