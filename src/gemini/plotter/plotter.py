#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gemini.utils import Logger

#===============================================================================


#===============================================================================

class PlotterOpt(Logger):
	''' Plotter  for optimizations with Gemini
	'''
	def __init__(self, dimension=2, model='gemini'):
		Logger.__init__(self, 'PlotterOpt', verbosity=2)
		self.dimension = dimension
		self.model = model

		self.fig, self.axes = plt.subplots(nrows=2,
										   ncols=3,
										   figsize=(15, 8),
								)
		self.axes[1, -1].set_visible(False)
		self.axes[1, -2].set_visible(False)


	def _set_colors(self):
		return None

	def plot_monitor(self, iteration, X, Y, cheap_Z, exp_Z,
					 params_cheap, params_exp,
					 objs_cheap, objs_exp,
					 ratio,
					 vals_0g_mesh=None, vals_1g_mesh=None,
				):

		for ax in self.axes.flatten():
			ax.clear()

		print(params_cheap.shape)
		print(params_exp.shape)

		# plot expensive surface
		contours = self.axes[0, 0].contour(X, Y, exp_Z, 3, colors='black')
		self.axes[0, 0].clabel(contours, inline=True, fontsize=8)
		self.axes[0, 0].imshow(exp_Z, extent=[-3, 3, -3, 3], origin='lower', cmap='coolwarm', alpha=0.5)
		self.axes[0, 0].set_title('Expensive surface')

		# plot the expensive evaluations
		self.axes[0, 0].plot(params_exp[:, 0], params_exp[:, 1],
							 ls='', marker='o', alpha=0.4, label='Observations')
		self.axes[0, 0].plot(params_exp[-2:, 0], params_exp[-2:, 1],
							 ls='', marker='D', label='Last observations')

		# plot the cheap surface
		contours = self.axes[1, 0].contour(X, Y, cheap_Z, 3, colors='black')
		self.axes[1, 0].clabel(contours, inline=True, fontsize=8)
		self.axes[1, 0].imshow(cheap_Z, extent=[-3, 3, -3, 3], origin='lower', cmap='coolwarm', alpha=0.5)
		self.axes[1, 0].set_title('Cheap surface')

		# plot the cheap evaluatiosn
		self.axes[1, 0].plot(params_cheap[:, 0], params_cheap[:, 1],
							 ls='', marker='o', alpha=0.4)
		self.axes[1, 0].plot(params_cheap[-2*ratio:, 0], params_cheap[-2*ratio:, 1],
							 ls='', marker='D')


		if iteration >= 1:
			# plot gemini acquisition (exploit)
			contours = self.axes[0, 1].contour(X, Y, vals_0g_mesh, 3, colors='black')
			self.axes[0, 1].clabel(contours, inline=True, fontsize=8)
			self.axes[0, 1].imshow(vals_0g_mesh, extent=[-3, 3, -3, 3], origin='lower', cmap='coolwarm', alpha=0.5)
			self.axes[0, 1].set_title('Gemini acquisition (exploit)')

			# plot gemini acquisition (explore)
			contours = self.axes[0, 2].contour(X, Y, vals_1g_mesh, 3, colors='black')
			self.axes[0, 2].clabel(contours, inline=True, fontsize=8)
			self.axes[0, 2].imshow(vals_1g_mesh, extent=[-3, 3, -3, 3], origin='lower', cmap='coolwarm', alpha=0.5)
			self.axes[0, 2].set_title('Gemini acquisition (explore)')

		self.axes[0, 0].legend(fontsize=8, loc='lower left', ncol=2)
		plt.tight_layout()
		plt.pause(2.0)



class PlotterReg(Logger):
	def __init__(self, dimension=1 , model_type='gemini'):
		''' Plottig module to monitor the training of gemini and bnn
		Can be used for examples with 1d or 2d features
		'''
		Logger.__init__(self, 'PlotterReg', verbosity=2)
		self.dimension  = dimension
		self.model_type = model_type
		self._set_colors()
		plt.ion()

		if self.dimension == 1:
			self.fig, self.axes = plt.subplots(nrows = 1,
											   ncols = 2,
											   figsize = (12, 4),
											   gridspec_kw={'width_ratios': [1, 2]})
		elif self.dimension > 1:
			message = f'Plotting not supported for dimension : {self.dimension}'
			self.logger.log(message, 'FATAL')


	def _set_colors(self):
		'''
		cheap surface    --> #79b473 bud-green
		exp surface      --> #005E7C blue sapphire
		cheap prediction --> #f19a3e deep saffron
		exp prediction   --> #403233 black coffee
		'''
		self.target_colors = ['#79b473', '#005E7C']
		self.pred_colors   = ['#403233', '#f19a3e']


	def plot_monitor(self, all_epochs, fast_losses, slow_losses,
					 valid_slow_raw, valid_slow_pred_raw,
					 train_slow_raw, train_slow_pred_raw,
					 valid_fast_raw, valid_fast_pred_raw,
					 valid_fast_features, valid_fast_targets,
					 train_fast_, train_fast_targets,
					 valid_slow_features, valid_slow_targets,
					 train_slow_features, train_slow_targets,
					 valid_std_pred_fast, valid_std_pred_slow):

		for ax in self.axes:
			ax.clear()


		self.axes[0].plot(valid_slow_raw, valid_slow_pred_raw, ls = '', marker = 'X', markersize = 6, color=self.target_colors[1],  label = 'validation')
		self.axes[0].plot(train_slow_raw, train_slow_pred_raw, ls = '', marker = 'o', markersize = 6, color=self.target_colors[0], label = 'training')
		if len(valid_slow_raw) > 0:
			all = np.concatenate((valid_slow_raw, valid_slow_pred_raw, train_slow_raw, train_slow_pred_raw), axis = 0)
		else:
			all = np.concatenate((train_slow_raw, train_slow_pred_raw), axis=0)
		min_, max_ = np.amin(all), np.amax(all)
		self.axes[0].plot([min_, max_], [min_, max_], color = 'k')
		self.axes[0].set_xlabel('True exp targets [a.u.]')
		self.axes[0].set_ylabel('Predicted exp targets [a.u.]')
		self.axes[0].legend()


		# plot the fast surface
		indices_fast = np.argsort(valid_fast_features, axis = 0).flatten()
		self.axes[1].plot(valid_fast_features[indices_fast], valid_fast_targets[indices_fast], color = self.target_colors[0],  ls='--', lw=3, label = 'Cheap response surface')
		self.axes[1].plot(train_fast_features, train_fast_targets, color = 'k', marker = 'o', ls = '', markersize = 8)
		self.axes[1].plot(train_fast_features, train_fast_targets, color = self.target_colors[0], marker = 'o', ls = '', markersize = 5, label = 'cheap training points')

		# # plot the fast prediction
		if valid_fast_features is not None:
			self.axes[1].plot(valid_fast_features[indices_fast], valid_fast_pred_raw[indices_fast], color = self.pred_colors[0], lw=3, label = 'Cheap prediction mu')
			self.axes[1].fill_between(np.squeeze(valid_fast_features[indices_fast]),
								np.squeeze(valid_fast_pred_raw[indices_fast] - valid_std_pred_fast[indices_fast]),
								np.squeeze(valid_fast_pred_raw[indices_fast] + valid_std_pred_fast[indices_fast]),
								color = self.pred_colors[0], alpha = 0.2, label='Cheap prediction sigma')

		# # plot the slow surface
		if valid_slow_features is not None:
			indices_slow = np.argsort(valid_slow_features, axis = 0).flatten()
			self.axes[1].plot(valid_slow_features[indices_slow], valid_slow_targets[indices_slow], color = self.target_colors[1], ls='--', lw=3, label = 'Expensive response surface')
		self.axes[1].plot(train_slow_features, train_slow_targets, color = 'k', marker = 'o', ls = '', markersize = 8)
		self.axes[1].plot(train_slow_features, train_slow_targets, color = self.target_colors[1], marker = 'o', ls = '', markersize = 5, label = 'exp training points')
		# # plot the slow prediction
		if valid_slow_features is not None:
			self.axes[1].plot(valid_slow_features[indices_slow], valid_slow_pred_raw[indices_slow], color = self.pred_colors[1], lw=3, label = 'Exp prediction mu')
			self.axes[1].fill_between(np.squeeze(valid_slow_features[indices_slow]),
								np.squeeze(valid_slow_pred_raw[indices_slow] - valid_std_pred_slow[indices_slow]),
								np.squeeze(valid_slow_pred_raw[indices_slow] + valid_std_pred_slow[indices_slow]),
								color = self.pred_colors[1], alpha = 0.2, label='Exp prediction sigma')

		self.axes[1].set_ylabel('Property [a.u.]')
		self.axes[1].set_xlabel('Condition [a.u.]')
		self.axes[1].legend(fontsize = 8, loc='lower left', ncol=3)

		plt.tight_layout()
		plt.pause(0.0005)
