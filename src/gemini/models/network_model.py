#!/usr/bin/env python

import os, sys
import pickle
import numpy as np
import tensorflow.compat.v1 as tf

#===============================================================================

from gemini.data_transformer import DataTransformer
from gemini.metrics import Metrics
from gemini.utils import Logger

#==============================================================================

class NetworkModel(Logger, DataTransformer):

	def __init__(self,
				 graph=None,
				 scope='model',
				 features_shape=1,
				 feature_scaling='normalization',
			   	 targets_shape=1,
				 target_scaling='standardization',
				 batch_size=100,
				 es_patience=None,
				 learning_rate=1e-4,
				 learning_rate_bias=1e-4,
				 learning_rate_latent=1e-3,
				 max_epochs=10**9,
				 pred_int=200,
				 skip_es=0,
				 layer='tf_dense',
				 train_metrics=['r2', 'mae', 'rmse', 'pearson', 'spearman'],
				 valid_metrics=['r2', 'mae', 'rmse', 'pearson', 'spearman'],
				 **kwargs,
			):
		Logger.__init__(self, 'NetworkModel', verbosity=2)
		DataTransformer.__init__(self)
		for key, val in locals().copy().items():
			if key in ['self', 'kwargs']: continue
			setattr(self, key, val)
		for key, val in kwargs.items():
			setattr(self, key, val)
		self.metrics = Metrics()

		self._is_graph_constructed = False
		self.graph = graph or tf.Graph()

	def check_device(self):
		# check for the prefered device - gpu or cpu
		if self.device=='gpu':
			assert(tf.test.is_built_with_cuda()), print('@@\tTF not built with CUDA')
			assert(len(tf.config.list_physical_devices('GPU'))>=1), print('GPU not found')
			print(tf.config.list_physical_devices('GPU'))
		elif self.device =='cpu':
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		else:
			raise NotImplementedError

	def register_hyperparams(self, hyperparams):
		if type(hyperparams) == dict:
			for key, val in hyperparams.items():
				setattr(self, key, val)
		else:
			self.log('User hyperparameters not understood, must provide key-value pairs', 'FATAL')

	def generator(self, features, targets):
		batch_idxs = np.random.randint(low=0, high=features.shape[0], size=self.batch_size)
		batch_features = features[batch_idxs]
		batch_targets = targets[batch_idxs]
		return batch_features, batch_targets


	# restore
	def restore(self, model_path):
		if not self._is_graph_constructed: self.build_inference()
		self.sess  = tf.Session(graph = self.graph)
		self.saver = tf.train.Saver()
		try:
			self.saver.restore(self.sess, f'{model_path}/model.ckpt')
			self.stats_dict = pickle.load(open(f'{model_path}/stats_dict.pkl', 'rb'))
			return True
		except AttributeError:
			return False

	# save
	def save_model(self, model_path):
		if model_path is not None:
			# save model parameters
			self.saver.save(self.sess, f'{model_path}/model.ckpt')
			# save training dataset stats
			pickle.dump(self.stats_dict, open(f'{model_path}/stats_dict.pkl', 'wb'))
		else:
			pass
