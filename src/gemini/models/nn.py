#!/usr/bin/env python

import time
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#==============================================================================

from gemini.metrics import Metrics
from gemini.models import NetworkModel
from gemini.utils import ACT_FUNCS, LAYERS, get_args

#==============================================================================

class NeuralNetwork(NetworkModel):

	def __init__(self, graph=None, scope='bnn',
			act_hidden='leaky_relu',
			act_out='linear',
			batch_size=50,
			depth=3,
			hidden=48,
			layer='tf_dense',
			device='cpu',
			reg=1e-3,
			user_hyperparams={},
			**kwargs,
		):
		super().__init__(**get_args(**locals()))
		self.register_hyperparams(user_hyperparams)
		self.check_device()

	def build_graph(self):
		layer = LAYERS[self.layer]
		act_hidden = ACT_FUNCS[self.act_hidden]
		act_out = ACT_FUNCS[self.act_out]

		with self.graph.as_default():
			self.tf_x = tf.placeholder(tf.float32, [self.batch_size, self.features_shape])
			self.tf_y = tf.placeholder(tf.float32, [self.batch_size, self.targets_shape])

		with self.graph.as_default():
			self.layers = [layer(self.hidden, activation=act_hidden) for _ in range(self.depth)]
			self.layers.append(layer(2 * self.targets_shape, activation=act_out))
			self.neural_net = tf.keras.Sequential(self.layers)
			pred = self.neural_net(self.tf_x)
			self.loc, self.scale = tf.split(pred, 2, axis=1)
			self.scale = tf.nn.sigmoid(self.scale) + 1e-1
			self.y_pred = tfd.Normal(loc=self.loc, scale=self.scale)

	def build_inference(self):
		self._is_graph_constructed = True
		with self.graph.as_default():
			self.elbo  = - tf.reduce_mean(self.y_pred.log_prob(self.tf_y))
			self.elbo += self.reg * (tf.reduce_mean(self.neural_net.losses) / self.batch_size)
			self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.train_op = self.optim.minimize(self.elbo)
			self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			self.sess = tf.Session(graph=self.graph)
			with self.sess.as_default():
				self.sess.run(self.init_op)

	def train(self, train_features, train_targets,
			  valid_features, valid_targets,
			  model_path=None, compute_loss=False, plot=False):
		''' If model_path is None, model is not saved
		'''


		if not self._is_graph_constructed: self.build_inference()

		self.dataset_stats(train_features, train_targets,
						   None, None)

		train_features_scaled = self.get_scaled(train_features, 'features', self.feature_scaling, 'slow')
		valid_features_scaled = self.get_scaled(valid_features, 'features', self.feature_scaling, 'slow')
		train_targets_scaled = self.get_scaled(train_targets, 'targets', self.target_scaling, 'slow')
		valid_targets_scaled = self.get_scaled(valid_targets, 'targets', self.target_scaling, 'slow')

		train_errors = []
		valid_errors = []
		losses = []
		all_epochs = []
		pred_epochs = []
		start_time = time.time()

		with self.graph.as_default():

			with self.sess.as_default():
				self.saver = tf.train.Saver()

				for epoch in range(self.max_epochs):
					all_epochs.append(epoch)
					train_features_batch, train_targets_batch = self.generator(train_features_scaled, train_targets_scaled)

					if compute_loss:
						loss, = self.sess.run([self.elbo], feed_dict={self.tf_x: train_features_batch, self.tf_y: train_targets_batch})
						losses.append(loss)

					_, = self.sess.run([self.train_op], feed_dict={self.tf_x: train_features_batch, self.tf_y: train_targets_batch})

					if epoch % self.pred_int == 0:
						pred_epochs.append(epoch)

						# predict validation set
						pred_loc, valid_scale = self.run_prediction(valid_features_scaled)
						valid_loc = self.get_raw(pred_loc, 'targets', self.target_scaling, 'slow')
						errors = self.metrics(valid_targets,
											  valid_loc,
											  self.valid_metrics)
						valid_errors.append(errors)

						# predict training set
						pred_loc, train_scale = self.run_prediction(train_features_scaled)
						train_loc = self.get_raw(pred_loc, 'targets', self.target_scaling, 'slow')
						errors = Metrics()(train_targets,
						           		   train_loc,
										   self.train_metrics)
						train_errors.append(errors)

						# save the model if we have just seen the best performance
						if model_path != None:
							os.makedirs(model_path, exist_ok = True)
							best_index = self.metrics.get_best_index('rmse', valid_errors)
							if best_index == len(valid_errors) - 1:
								self.log('Saving model parameters', 'INFO')
								self.saver.save(self.sess, f'{model_path}/model.ckpt')
						# check early stopping criteria
						if self.metrics.early_stopping('rmse', valid_errors, self.es_patience):
							self.log('Reached early stopping criteria', 'INFO')
							break
						# print results
						print(f'EPOCH: {epoch}   TRAIN_R2: {train_errors[-1]["r2"]}   VALID_R2: {valid_errors[-1]["r2"]}')

				train_results = self.metrics.get_all_best_metric(self.train_metrics, train_errors)
				valid_results = self.metrics.get_all_best_metric(self.valid_metrics, valid_errors)

		return train_results, valid_results


	def run_prediction(self, features, scaled = True):
		if not scaled:
			if not hasattr(self, 'stats_dict'):
				self.log('You must specify the statistics of the dataset to transform the features', 'FATAL')
			else:
				features = self.get_scaled(features, 'features', self.feature_scaling, 'slow')

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
			predic_mu    = self.sess.run( self.loc, feed_dict = {self.tf_x: X_batch})
			predic_std   = self.sess.run(self.scale, feed_dict={self.tf_x: X_batch})
			pred_mu[start:stop] = predic_mu[:size]
			pred_std[start:stop] = predic_std[:size]

		# scale the predictions
		pred_mu_raw = self.get_raw(pred_mu, 'targets', self.target_scaling, 'slow')

		return pred_mu, pred_std
