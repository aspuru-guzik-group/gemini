#!/usr/bin/env python


import time
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#==============================================================================

from gemini.models import NetworkModel
from gemini.models import OPT_HPARAMS as OPTIMAL_HPARAMS
from gemini.utils import ACT_FUNCS, LAYERS, get_args
from gemini.utils import parse_feature_vars, parse_target_vars, parse_latent_vars
from gemini.metrics import Metrics
from gemini.plotter import PlotterReg
#==============================================================================

class Gemini(NetworkModel):

	OPT_HPARAMS = OPTIMAL_HPARAMS

	def __init__(self,
	 			 graph=None,
				 scope='gemini',
				 user_hyperparams={},
				 **kwargs,
			):
		super().__init__(**get_args(**locals()))
		self.register_hyperparams(self.OPT_HPARAMS)
		self.register_hyperparams(user_hyperparams)
		self.check_device()

		self.train_metrics, self.valid_metrics = ['r2', 'rmse'], ['r2','rmse']


	def build_graph(self):
		layer = LAYERS[self.layer]
		act_fbias = ACT_FUNCS[self.act_fbias]
		act_fbias_out = ACT_FUNCS[self.act_fbias_out]
		act_latent = ACT_FUNCS[self.act_latent]
		act_latent_out = ACT_FUNCS[self.act_latent_out]
		act_tbias = ACT_FUNCS[self.act_tbias]
		act_tbias_out = ACT_FUNCS[self.act_tbias_out]

		with self.graph.as_default():
			self.tf_x_fast = tf.placeholder(tf.float32, [self.batch_size, self.features_shape])
			self.tf_x_slow = tf.placeholder(tf.float32, [self.batch_size, self.features_shape])
			self.tf_y_fast = tf.placeholder(tf.float32, [self.batch_size, self.targets_shape])
			self.tf_y_slow = tf.placeholder(tf.float32, [self.batch_size, self.targets_shape])

			with tf.name_scope('feature_slow'):
				if self.act_fbias == 'lambda':
					layers = [LAYERS['lambda'](lambda y: y)]
				else:
					layers = [tf.keras.layers.BatchNormalization()]
					for _ in range(self.depth_fbias):
						layers.append(layer(self.features_shape, activation=act_fbias,
								kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
								kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_bias),
								bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
								bias_regularizer=tf.keras.regularizers.l2(l=self.reg_bias)))
						layers.append(tf.keras.layers.BatchNormalization())
					layers.append(layer(self.features_shape, activation=act_fbias_out,
							kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
							kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_bias),
							bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
							bias_regularizer=tf.keras.regularizers.l2(l=self.reg_bias)))
				self.nn_feature_slow = tf.keras.Sequential(layers)

			with tf.name_scope('latent'):
				layers = [tf.keras.layers.BatchNormalization()]
				for _ in range(self.depth_latent):
					layers.append(layer(self.hidden_latent, activation=act_latent, kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_latent)))
					layers.append(tf.keras.layers.BatchNormalization())
				layers.append(layer(2 * self.targets_shape, activation=act_latent_out))
				self.nn_latent = tf.keras.Sequential(layers)

			with tf.name_scope('target_slow'):
				if self.act_tbias == 'lambda':
					layers = [LAYERS['lambda'](lambda y: y)]
				else:
					layers = [tf.keras.layers.BatchNormalization()]
					for _ in range(self.depth_tbias):
						layers.append(layer(self.hidden_tbias, activation=act_tbias,
								kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
								kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_bias),
								bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
								bias_regularizer=tf.keras.regularizers.l2(l=self.reg_bias)))
						layers.append(tf.keras.layers.BatchNormalization())
					layers.append(layer(2 * self.targets_shape, activation=act_tbias_out,
							kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
							kernel_regularizer=tf.keras.regularizers.l2(self.reg_bias),
							bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
							bias_regularizer=tf.keras.regularizers.l2(l=self.reg_bias)))
				self.nn_targets_slow = tf.keras.Sequential(layers)

			# register fast stream
			pred_fast = self.nn_latent( self.tf_x_fast )
			self.loc_fast, self.scale_fast = tf.split(pred_fast, 2, axis=1)
			self.scale_fast = tf.nn.sigmoid(self.scale_fast) + 1e-2
			self.y_pred_fast = tfd.Normal(loc=self.loc_fast, scale=self.scale_fast)

			if np.logical_and(self.act_fbias == 'lambda', self.act_tbias == 'lambda'):
				# transfer learning
				pred_slow = self.nn_latent( self.tf_x_slow )
			else:
				# gemini
				features_slow_perturbed = self.tf_x_slow + self.nn_feature_slow( self.tf_x_slow )
				pred_slow = self.nn_latent( features_slow_perturbed ) + self.nn_targets_slow( self.nn_latent( features_slow_perturbed ) )
			self.loc_slow, self.scale_slow = tf.split(pred_slow, 2, axis=1)
			self.scale_slow = tf.nn.sigmoid(self.scale_slow) + 1e-1
			self.y_pred_slow = tfd.Normal(loc=self.loc_slow, scale=self.scale_slow)

	def build_inference(self):
		self._is_graph_constructed = True
		with self.graph.as_default():

			self.kl_fast = - tf.reduce_mean(self.y_pred_fast.log_prob(self.tf_y_fast))
			self.kl_slow = - tf.reduce_mean(self.y_pred_slow.log_prob(self.tf_y_slow))

			self.reg_latent = tf.reduce_mean(self.nn_latent.losses)
			self.reg_bias   = tf.reduce_mean(self.nn_feature_slow.losses) + tf.reduce_mean(self.nn_targets_slow.losses)
			self.reg_loss   = self.reg_latent + self.reg_bias

			self.loss_fast = self.kl_fast + self.reg_latent
			self.loss_slow = self.kl_slow + self.reg_bias

			self.optim_both = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.train_op_both = self.optim_both.minimize(self.loss_slow + self.coeff_both * self.loss_fast)

			self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			self.sess = tf.Session(graph=self.graph)
			with self.sess.as_default():
				self.sess.run(self.init_op)


	def train(self, train_fast_features, train_fast_targets,
			  valid_fast_features, valid_fast_targets,
			  train_slow_features, train_slow_targets,
			  valid_slow_features, valid_slow_targets,
			  model_path, compute_loss=False, plot=False):

		if not self._is_graph_constructed: self.build_inference()

		# compute stats dict
		self.dataset_stats(train_slow_features, train_slow_targets,
						   train_fast_features, train_fast_targets)

		valid_args = [valid_fast_features, valid_fast_targets, valid_slow_features, valid_slow_targets]
		if all(isinstance(v, type(None)) for v in valid_args):
			is_valid = False
		else:
			is_valid = True
			self.valid_fast_features_scaled = self.get_scaled(valid_fast_features, 'features', self.feature_scaling, 'fast')
			self.valid_fast_targets_scaled  = self.get_scaled(valid_fast_targets, 'targets', self.target_scaling, 'fast')
			self.valid_slow_features_scaled = self.get_scaled(valid_slow_features, 'features', self.feature_scaling, 'slow')
			self.valid_slow_targets_scaled  = self.get_scaled(valid_slow_targets,  'targets', self.target_scaling, 'slow')

		self.train_fast_features_scaled = self.get_scaled(train_fast_features, 'features', self.feature_scaling, 'fast')
		self.train_fast_targets_scaled  = self.get_scaled(train_fast_targets, 'targets', self.target_scaling, 'fast')
		self.train_slow_features_scaled = self.get_scaled(train_slow_features, 'features', self.feature_scaling, 'slow')
		self.train_slow_targets_scaled  = self.get_scaled(train_slow_targets, 'targets', self.target_scaling, 'slow')


		#====================================

		train_fast_errors = list()
		train_slow_errors = list()
		valid_fast_errors = list()
		valid_slow_errors = list()

		valid_slow_feature_errors = list()
		valid_fast_feature_errors = list()
		valid_slow_latent_errors = list()
		valid_fast_latent_errors = list()

		train_slow_feature_errors = list()
		train_fast_feature_errors = list()
		train_slow_latent_errors = list()
		train_fast_latent_errors = list()

		fast_losses       = []
		slow_losses       = []
		all_epochs        = list()
		pred_epochs       = list()
		start_time        = time.time()

		#====================================

		with self.graph.as_default():
			if plot:
				plotter = PlotterReg(self.features_shape,
									 'gemini')

			with self.sess.as_default():
				self.saver = tf.train.Saver()

				loss_fast, loss_slow = 1., 1.

				for epoch in range(self.max_epochs):
					all_epochs.append(epoch)

					if compute_loss:

						train_fast_features_batch, train_fast_targets_batch = self.generator(self.train_fast_features_scaled, self.train_fast_targets_scaled)
						train_slow_features_batch, train_slow_targets_batch = self.generator(self.train_slow_features_scaled, self.train_slow_targets_scaled)

						loss_fast = self.sess.run(self.loss_fast, feed_dict = {self.tf_x_fast: train_fast_features_batch,
																			   self.tf_y_fast: train_fast_targets_batch})
						loss_slow = self.sess.run(self.loss_slow, feed_dict = {self.tf_x_slow: train_slow_features_batch,
																			   self.tf_y_slow: train_slow_targets_batch})

					else:
						loss_fast = 0.
						loss_slow = 0.
					fast_losses.append(loss_fast)
					slow_losses.append(loss_slow)



					if epoch % 10 == 0:
						train_fast_features_batch, train_fast_targets_batch = self.generator(self.train_fast_features_scaled, self.train_fast_targets_scaled)
						train_slow_features_batch, train_slow_targets_batch = self.generator(self.train_slow_features_scaled, self.train_slow_targets_scaled)

					if compute_loss:
						__, reg_loss = self.sess.run([self.train_op_both, self.reg_loss], feed_dict={self.tf_x_slow: train_slow_features_batch,
																									self.tf_y_slow: train_slow_targets_batch,
																									self.tf_x_fast: train_fast_features_batch,
																									self.tf_y_fast: train_fast_targets_batch,})

					else:
						__, = self.sess.run([self.train_op_both], feed_dict={self.tf_x_slow: train_slow_features_batch,
																			self.tf_y_slow: train_slow_targets_batch,
																			self.tf_x_fast: train_fast_features_batch,
																			self.tf_y_fast: train_fast_targets_batch,})


					if epoch % self.pred_int == 0:

						pred_epochs.append(epoch)

						if is_valid:

							# make a prediction on the fast validation set
							valid_fast_pred_raw, valid_std_pred_fast = self.run_prediction(self.valid_fast_features_scaled, 'fast')
							valid_fast_raw     = self.get_raw(self.valid_fast_targets_scaled, 'targets', self.target_scaling, 'fast')
							errors = self.metrics(valid_fast_raw,
												 valid_fast_pred_raw,
												 self.valid_metrics)
							valid_fast_errors.append(errors)

							# make a prediction on the slow validation set
							valid_slow_pred_raw, valid_std_pred_slow = self.run_prediction(self.valid_slow_features_scaled, 'slow')
							valid_slow_raw      = self.get_raw(self.valid_slow_targets_scaled, 'targets', self.target_scaling, 'slow')
							errors = self.metrics(valid_slow_raw,
												 valid_slow_pred_raw,
												 self.valid_metrics)
							valid_slow_errors.append(errors)

						else:
							valid_slow_raw = []
							valid_slow_pred_raw = []
							valid_fast_raw = []
							valid_fast_pred_raw = []
							valid_std_pred_fast = []
							valid_std_pred_slow = []

						# make prediction on the fast train set
						train_fast_pred_raw, train_std_pred_fast = self.run_prediction(self.train_fast_features_scaled, 'fast')
						train_fast_raw     = self.get_raw(self.train_fast_targets_scaled, 'targets', self.target_scaling, 'fast')
						errors = self.metrics(train_fast_raw,
											 train_fast_pred_raw,
											 self.train_metrics)
						train_fast_errors.append(errors)

						# make a prediction on the slow train set
						train_slow_pred_raw, train_std_pred_slow = self.run_prediction(self.train_slow_features_scaled, 'slow')
						train_slow_raw     = self.get_raw(self.train_slow_targets_scaled, 'targets', self.target_scaling, 'slow')
						errors = self.metrics(train_slow_raw,
											 train_slow_pred_raw,
											 self.train_metrics)
						train_slow_errors.append(errors)

						# save the model every prediction up until self.skip_es*self.pred_int epochs
						if self.es_patience is not None and is_valid:
							if epoch <= self.skip_es*self.pred_int:
								self.save_model(model_path)
								self.log('Saving model parameters', 'INFO')
							# start counting early stopping patience
							else:
								best_index = self.metrics.get_best_index('rmse', valid_slow_errors[self.skip_es:])
								if best_index == len(valid_slow_errors[self.skip_es:]) - 1:
									self.save_model(model_path)
									self.log('Saving model parameters', 'INFO')

								# check early stopping, but only starting from self.skip_es*self.pred_int epochs
								if self.metrics.early_stopping('rmse', valid_slow_errors[self.skip_es:], self.es_patience):
									self.log('Reached early stopping criteria', 'INFO')
									# record the maximum number of epochs reached as model attribute
									self.max_epochs = epoch
									break
						else:
							# if no early stopping, save only on the last prediction epoch
							if epoch in range(self.max_epochs-self.pred_int, self.max_epochs):
								self.save_model(model_path)
								self.log('Saving model parameters', 'INFO')

						error_kind = 'r2'
						if is_valid:
							print(f'EPOCH: {epoch}\tTRAIN_FAST_ERROR: {train_fast_errors[-1][error_kind]}\tTRAIN_SLOW_ERROR: {train_slow_errors[-1][error_kind]}\tVALID_FAST_ERROR: {valid_fast_errors[-1][error_kind]}\tVALID_SLOW_ERROR: {valid_slow_errors[-1][error_kind]}')
						else:
							print(f'EPOCH: {epoch}\tTRAIN_FAST_ERROR: {train_fast_errors[-1][error_kind]}\tTRAIN_SLOW_ERROR: {train_slow_errors[-1][error_kind]}')

						# plot iteration
						if plot:
							plotter.plot_monitor(all_epochs, fast_losses, slow_losses,
												 valid_slow_raw, valid_slow_pred_raw,
												 train_slow_raw, train_slow_pred_raw,
												 valid_fast_raw, valid_fast_pred_raw,
												 valid_fast_features, valid_fast_targets,
												 train_fast_features, train_fast_targets,
												 valid_slow_features, valid_slow_targets,
												 train_slow_features, train_slow_targets,
												 valid_std_pred_fast, valid_std_pred_slow)

				self.log(f'Finished training: {epoch+1} epochs in {round(time.time()-start_time, 6)} seconds', 'INFO')

				# parse variables
				vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
				feature_vars = [v for v in vars if 'sequential/' in v.name]
				latent_vars  = [v for v in vars if 'sequential_1/' in v.name]
				target_vars  = [v for v in vars if 'sequential_2/' in v.name]

				parsed_feature_vars = parse_feature_vars(feature_vars)
				parsed_latent_vars  = parse_latent_vars(latent_vars)
				parsed_target_vars  = parse_target_vars(target_vars)

				parsed_vars = {'feature': parsed_feature_vars, 'latent': parsed_latent_vars, 'target': parsed_target_vars}

				if is_valid:
					train_results = self.metrics.get_all_best_metric(self.train_metrics, train_slow_errors)
					valid_results = self.metrics.get_all_best_metric(self.valid_metrics, valid_slow_errors)
					return train_results, valid_results, parsed_vars

				else:
					# return train errors on the final epoch for optimization
					return train_slow_errors[-1], {}, parsed_vars


	def run_prediction(self,
					  features,
					  evaluator='slow',
					  scaled=False,
					  return_unscaled=False
			):
		''' predictions with Gemini
		'''
		if not scaled:
			if not hasattr(self, 'stats_dict'):
				self.log('You must specify the statistics of the dataset to transform the features', 'FATAL')
			else:
				features = self.get_scaled(features, 'features', self.feature_scaling, evaluator)

		pred_mu     = np.empty((features.shape[0], self.targets_shape))
		pred_std    = np.empty((features.shape[0], self.targets_shape))
		resolution  = divmod(features.shape[0], self.batch_size)
		res         = [self.batch_size for i in range(resolution[0])]
		res.append(resolution[1])
		res     = list(filter((0).__ne__, res))
		res_cu  = [i*self.batch_size for i in range(len(res))]
		res_stp = [res_cu[_]+re for _, re in enumerate(res)]

		for batch_iter, size in enumerate(res):
			start, stop  = res_cu[batch_iter], res_stp[batch_iter]
			X_batch = None
			if size == self.batch_size:
				X_batch = features[start: stop]
			elif size != self.batch_size:
				X_batch = np.concatenate((features[start: stop], np.random.choice(features[:, 0], size = (self.batch_size - size, features.shape[1]))), axis = 0)
			if evaluator == 'fast':
				predic_mu, predic_std = self.sess.run([self.loc_fast, self.scale_fast], feed_dict = {self.tf_x_fast: X_batch})
			elif evaluator == 'slow':
				predic_mu, predic_std = self.sess.run([self.loc_slow, self.scale_slow], feed_dict = {self.tf_x_slow: X_batch})

			pred_mu[start:stop] = predic_mu[:size]
			pred_std[start:stop] = predic_std[:size]

		if not return_unscaled:
			pred_mu_raw = self.get_raw(pred_mu, 'targets', self.target_scaling, evaluator)
		else:
			pred_mu_raw = pred_mu

		return pred_mu_raw, pred_std
