#!/usr/bin/env python

import os, sys
import numpy as np
import tensorflow as tf

#===============================================================================

from gemini.data_transformer import DataTransformer
from gemini.datasets         import Dataset
from gemini.utils            import Logger, get_args

#===============================================================================

class Base(DataTransformer, Logger):
	'''
	Base methods for gemini
	'''

	def __init__(self,
				model_type='gemini',
				model_path='./.tmp_models',
				features_shape=1,
				targets_shape=1,
				**kwargs,
		):

		DataTransformer.__init__(self)
		Logger.__init__(self, 'Base', verbosity=2)
		self.model_type = model_type
		self.model_path = model_path
		self.is_opt = False
		self.is_trained = False
		for key, val in locals().copy().items():
			if key in ['self', 'kwargs']: continue
			setattr(self, key, val)
		for key, val in kwargs.items():
			setattr(self, key, val)


	def create_models(self, hyperparams={}):
		if self.model_type == 'nn':
			from gemini.models.nn import NeuralNetwork as Model
		elif self.model_type == 'bnn':
			from gemini.models.bnn import BayesianNeuralNetwork as Model
		elif self.model_type == 'gemini':
			from gemini.models.gemini import Gemini as Model
		else:
			self.log(f'Model type {self.model_type} not understood', 'FATAL')

		if self.num_folds > 1:
			self.cv_graphs = [tf.Graph() for fold in range(self.num_folds)]
			self.cv_models = [Model(graph=graph, user_hyperparams=hyperparams, features_shape=self.features_shape, targets_shape=self.targets_shape) for graph in self.cv_graphs]

		# make the full graph regardless
		self.full_graph = tf.Graph()
		self.full_model = Model(graph=self.full_graph,
								user_hyperparams=hyperparams,
								features_shape=self.features_shape,
								targets_shape=self.targets_shape,)

	def build_models(self):
		if self.cv_models is not None:
			for _, model in enumerate(self.cv_models):
				with self.cv_graphs[_].as_default():
					model.build_graph()

		with self.full_graph.as_default():
			self.full_model.build_graph()


	def load_dataset(self, name, alpha):
		self.dataset_object   = Dataset(name)
		self.features_shape = self.dataset_object.slow_features_shape
		self.targets_shape  = self.dataset_object.slow_targets_shape
		self.dataset = self.dataset_object.load_full(alpha = alpha)
		self.num_folds = len(self.dataset[0]['data']['train_slow_features'])

		return self.dataset


	def train_fold(self, fold_ix, fold):
		self.log(f'Training on CV fold {fold_ix}/{self.num_folds-1}', 'INFO')
		train_results, __, ___  = self.cv_models[fold_ix].train(fold['train_fast_features'], fold['train_fast_targets'],
																fold['valid_fast_features'], fold['valid_fast_targets'],
																fold['train_slow_features'], fold['train_slow_targets'],
																fold['valid_slow_features'], fold['valid_slow_targets'],
																model_path = self.model_path+f'/fold_{fold_ix}')
		max_epochs = self.cv_models[fold_ix].max_epochs

		return max_epochs

	def train(self,
			  train_slow_features,
			  train_slow_targets,
			  train_fast_features,
			  train_fast_targets,
			  valid_slow_features=None,
			  valid_slow_targets=None,
			  valid_fast_features=None,
			  valid_fast_targets=None,
			  num_folds=1,
			  user_hyperparams={},
			  plot=False):
		# check the shape of the features and targets and set shape
		if train_slow_features.shape[1] != train_fast_features.shape[1]:
			self.log('Cheap and expenisve parameters have differing dimensions', 'FATAL')
		else:
			self.features_shape = train_slow_features.shape[1]
		if train_slow_targets.shape[1] != train_fast_targets.shape[1]:
			self.log('Cheap and expenisve targets have differing dimensions', 'FATAL')
		else:
			self.targets_shape = train_slow_targets.shape[1]

		setattr(self, 'num_folds', num_folds)
		# check to see if the number of folds is appropriate
		if self.num_folds < 1:
			self.log('num folds argument must be a positive integer - skipping cross validation', 'WARNING')
			self.num_folds = 1
		if self.is_opt and self.num_folds == 1:
			self.log('Using Gemini for optimization without cross-validation is quicker but may over-estimate correlation', 'WARNING')

		self.cv_models = None
		# create and build the model(s)
		self.create_models(user_hyperparams)
		self.build_models()

		# make temporary directory for model weights
		if self.model_path is not None:
			os.makedirs(self.model_path, exist_ok=True)
		else:
			if self.is_opt:
				self.log('You must specify a directory to save model weights for use with optimization', 'FATAL')

		if self.num_folds > 1:
			# generate the folds
			folds = self._get_folds(train_slow_features, train_slow_targets,
									train_fast_features, train_fast_targets)
			cv_epochs = []
			# run the cross validation folds sequentially
			for fold_ix, fold in enumerate(folds):
				max_epochs = self.train_fold(fold_ix, fold)
				cv_epochs.append(self.cv_models[fold_ix].max_epochs)

			if self.is_opt and self.cv_models is not None:
				# estimate pearsons coefficient
				self.estimate_pearson_coeff(train_slow_features, train_slow_targets)
				# turn off early stopping for the full model
				self.full_model.es_patience = None
				# set max epochs for the full model as mean of cv folds
				self.full_model.max_epochs = int(np.mean(cv_epochs))

		# train on the entire set
		self.log('Training on the entire training set', 'INFO')
		train_results, valid_results, params = self.full_model.train(train_fast_features, train_fast_targets,
														 valid_fast_features, valid_fast_targets,
														 train_slow_features, train_slow_targets,
														 valid_slow_features, valid_slow_targets,
														 model_path = self.model_path+f'/full', plot=plot)

		# estimate the pearson coefficient if no cv
		if self.is_opt and not self.cv_models:
			self.estimate_pearson_coeff(train_slow_features, train_slow_targets)

		# set is trained
		self.is_trained = True
		# remove the temporary weights directory between optimization iterations
		if self.is_opt:
			os.system(f'rm -r {self.model_path}')


		return train_results, valid_results, params


	def _get_folds(self, train_slow_features, train_slow_targets,
				   train_fast_features, train_fast_targets):

		# only use max folds for optimization cross validation
		num_train_slow = train_slow_features.shape[0]
		if num_train_slow < self.num_folds:
			self.num_folds = num_train_slow
		else:
			self.num_folds = self.num_folds

		fold_sizes = []
		fold_sizes.extend([num_train_slow//self.num_folds+1 for _ in range(num_train_slow%self.num_folds)])
		fold_sizes.extend([num_train_slow//self.num_folds for _ in range(self.num_folds-len(fold_sizes))])

		indices = np.arange(num_train_slow)
		np.random.shuffle(indices)

		folds = []
		for fold_size in fold_sizes:
			train_slow_features_fold, train_slow_targets_fold = train_slow_features[indices[fold_size:]], train_slow_targets[indices[fold_size:]]
			valid_slow_features_fold, valid_slow_targets_fold = train_slow_features[indices[:fold_size]], train_slow_targets[indices[:fold_size]]

			fold = {'train_slow_features': train_slow_features_fold, 'train_slow_targets': train_slow_targets_fold,
					'valid_slow_features': valid_slow_features_fold, 'valid_slow_targets': valid_slow_targets_fold,
					'train_fast_features': train_fast_features, 'train_fast_targets': train_fast_targets,
					'valid_fast_features': train_fast_features, 'valid_fast_targets': train_fast_targets}

			folds.append(fold)
			indices = np.roll(indices, fold_size)
		return folds


	def load_cv_models(self, num_folds=None):
		if num_folds == None:
			num_folds = self.num_folds
		self.cv_models = [self.load_model(model_path=self.model_path+f'/fold_{fold_ix}') for fold_ix in range(num_folds)]

	def load_full_model(self):
		self.full_model = self.load_model(model_path=self.model_path+f'/full')

	def load_model(self, model_path=None, hyperparams={}):
		if model_path == None:
			model_path = self.model_path
		self.graph = tf.Graph()
		with self.graph.as_default():
			if self.model_type == 'bnn':
				from gemini.models.bnn import BayesianNeuralNetwork as Model
			elif self.model_type == 'gemini':
				from gemini.models.gemini import Gemini as Model
			else:
				raise NotImplementedError
			self.model = Model(graph=self.graph,
							   user_hyperparams=hyperparams,
							   features_shape=self.features_shape,
							   targets_shape=self.targets_shape,)

			self.model.build_graph()
			if self.model.restore(model_path):
				self.log('Successfully restored saved model', 'INFO')
				return self.model
			else:
				self.log('There was an issue restoring saved model', 'FATAL')


	# TODO: implement this bit
	def update_hyperparams(self, updated_hyperparams):
		return None

	def predict_cv(self, features, evaluator='slow', scaled=False):
		''' Returns the mean prediction over several cross-validation folds
		'''
		preds = []
		stds  = []
		for model in self.cv_models:
			mean_pred, std_pred = model.run_prediction(features, evaluator, scaled)
			preds.append(mean_pred)
			stds.append(std_pred)
		return np.mean(np.array(preds), axis=0), np.mean(np.array(std_pred), axis=0)

	def predict(self, features, evaluator='slow', scaled=False):
		''' Returns the prediction of the full model trained on all of the data
		'''
		if not hasattr(self, 'full_model'):
			self.load_full_model()
		mean_pred, std_pred = self.full_model.run_prediction(features, evaluator, scaled)
		return mean_pred, std_pred


if __name__ == '__main__':
	# # DEBUG:

	base = Base(model_type='gemini', model_path='./.tmp_models')
