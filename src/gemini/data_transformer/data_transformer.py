#!/usr/bin/env python

import numpy as np
import pickle
import json

#===============================================================================

class DataTransformer(object):
	'''
	Data scaling
	'''

	def __init__(self):
		pass

	def dataset_stats(self, train_slow_features, train_slow_targets,
					  train_fast_features, train_fast_targets):
		'''
		compute stats dictionary using only the training data
		'''

		features_shape     = [None, np.array(train_slow_features).shape[1]]
		targets_shape      = [None, np.array(train_slow_targets).shape[1]]

		mean_slow_features = np.mean(train_slow_features, axis = 0)
		std_slow_features  = np.std(train_slow_features, axis = 0)
		std_slow_features  = np.where(std_slow_features == 0., 1., std_slow_features)
		mean_slow_targets  = np.mean(train_slow_targets, axis = 0)
		std_slow_targets   = np.std(train_slow_targets, axis = 0)
		min_slow_features  = np.amin(train_slow_features, axis = 0)
		max_slow_features  = np.amax(train_slow_features, axis = 0)
		min_slow_targets   = np.amin(train_slow_targets, axis = 0)
		max_slow_targets   = np.amax(train_slow_targets, axis = 0)

		stats_dict_slow = {'features_shape': features_shape, 'targets_shape': targets_shape,
						  'mean_slow_features': mean_slow_features, 'std_slow_features': std_slow_features,
						  'mean_slow_targets': mean_slow_targets, 'std_slow_targets': std_slow_targets,
						  'min_slow_features': min_slow_features, 'max_slow_features': max_slow_features,
						  'min_slow_targets': min_slow_targets, 'max_slow_targets': max_slow_targets}

		if train_fast_features is not None:
			min_fast_features  = np.amin(train_fast_features, axis = 0)
			max_fast_features  = np.amax(train_fast_features, axis = 0)
			min_fast_targets   = np.amin(train_fast_targets, axis = 0)
			max_fast_targets   = np.amax(train_fast_targets, axis = 0)
			mean_fast_features = np.mean(train_fast_features, axis = 0)
			std_fast_features  = np.std(train_fast_features, axis = 0)
			std_fast_features  = np.where(std_fast_features == 0., 1., std_fast_features)
			mean_fast_targets  = np.mean(train_fast_targets, axis = 0)
			std_fast_targets   = np.std(train_fast_targets, axis = 0)

			stats_dict_fast = {'mean_fast_features': mean_fast_features, 'std_fast_features': std_fast_features,
							  'mean_fast_targets': mean_fast_targets, 'std_fast_targets': std_fast_targets,
							  'min_fast_features': min_fast_features, 'max_fast_features': max_fast_features,
							  'min_fast_targets': min_fast_targets, 'max_fast_targets': max_fast_targets}

			self.stats_dict = {**stats_dict_slow, **stats_dict_fast}
		else:
			self.stats_dict = stats_dict_slow

		return self.stats_dict


	def get_scaled(self, input, type, scaling, evaluator):
		''' Forward transformation
		'''
		if scaling == 'standardization':
			scaled_input = (input - self.stats_dict[f'mean_{evaluator}_{type}']) / self.stats_dict[f'std_{evaluator}_{type}']
		elif scaling == 'mean':
			scaled_input = input / self.stats_dict[f'mean_{evaluator}_{type}']
		elif scaling == 'same':
			scaled_input = input
		elif scaling == 'normalization':
			scaled_input =  (input - self.stats_dict[f'min_{evaluator}_{type}']) / (self.stats_dict[f'max_{evaluator}_{type}'] - self.stats_dict[f'min_{evaluator}_features'])

		return scaled_input


	def get_raw(self, input, type, scaling, evaluator):
		''' Reverse transformation
		'''
		if scaling == 'standardization':
			raw_input = input * self.stats_dict[f'std_{evaluator}_{type}'] + self.stats_dict[f'mean_{evaluator}_{type}']
		elif scaling == 'mean':
			raw_input = input * self.stats_dict[f'mean_{evaluator}_{type}']
		elif scaling == 'same':
			raw_input = input
		elif scaling == 'normalization':
			raw_input = (self.stats_dict[f'max_{evaluator}_{type}'] - self.stats_dict[f'min_{evaluator}_{type}']) * input + self.stats_dict[f'min_{evaluator}_{type}']

		return raw_input
