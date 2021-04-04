#!/usr/bin/env python

import pytest

import numpy as np

from gemini.data_transformer import DataTransformer
from gemini.models.nn import NeuralNetwork

#=============================================================================

def func(x):
    return np.exp(-2 *x) * np.cos(4 * np.pi * x)


def test_nn():
    '''
    Test training of BNN on analytic functions by interacting with BNN
    from the model level (lower level implementation)
    '''
    np.random.seed(100691)
    train_features = np.random.uniform(low=0, high=1, size=(5, 1))
    train_targets  = func(train_features)

    valid_features = np.random.uniform(low=0, high=1, size=(500, 1))
    valid_targets  = func(valid_features)

    net = NeuralNetwork()
    net.max_epochs = 1*10**3 # short training for the test
    net.build_graph()
    net.build_inference()
    train_results, valid_results = net.train(train_features, train_targets,
                                             valid_features, valid_targets,
                                            model_path=None, compute_loss=False)

    assert(type(train_results) == dict)
    assert(type(valid_results) == dict)

    features = np.random.uniform(0, 1, (10, 1))

    mean_pred, std_pred = net.run_prediction(features, scaled=False)

    assert(type(mean_pred) == np.ndarray)
    assert(type(std_pred) == np.ndarray)
    assert(mean_pred.shape == features.shape)
    assert(std_pred.shape == features.shape)


if __name__ == '__main__':
    test_nn()
