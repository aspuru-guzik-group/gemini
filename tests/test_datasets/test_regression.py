#!/usr/bin/env python

import pytest
import numpy as np

from gemini import Gemini
from gemini import __datasets__
from gemini.datasets import Dataset

#===============================================================================


def test_regression(dataset_name='photobleaching', model_type='gemini'):

    dataset = Dataset(name=dataset_name)

    indices = dataset.generate_emulator_indices(dataset.data, frac_test=0.2)

    cheap_data = dataset.load_emulator(eval='fast')
    exp_data   = dataset.load_emulator(eval='slow')

    assert(type(cheap_data) == dict)
    assert(type(exp_data) == dict)

    hyperparams = {'max_epochs': 1000}

    gemini = Gemini()

    res = gemini.train(exp_data['train_features'], exp_data['train_targets'],
                         cheap_data['train_features'], cheap_data['train_targets'],
                         user_hyperparams=hyperparams)

    train_results, valid_results, params = res

    assert(type(train_results)==dict)
    assert(type(valid_results)==dict)
    assert(type(params)==dict)


    # make predictions
    cheap_pred_mu, cheap_pred_std = gemini.predict(cheap_data['test_features'], 'fast')
    exp_pred_mu, exp_pred_std     = gemini.predict(exp_data['test_features'], 'slow')

    assert(type(cheap_pred_mu)==np.ndarray)
    assert(type(cheap_pred_std)==np.ndarray)
    assert(type(exp_pred_mu)==np.ndarray)
    assert(type(exp_pred_std)==np.ndarray)

    assert(cheap_pred_mu.shape==cheap_data['test_targets'].shape)
    assert(cheap_pred_std.shape==cheap_data['test_targets'].shape)
    assert(exp_pred_mu.shape==exp_data['test_targets'].shape)
    assert(exp_pred_std.shape==exp_data['test_targets'].shape)


if __name__ == '__main__':
    test_regression(dataset_name='photobleaching',
                    model_type='gemini')
