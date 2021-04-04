#!/usr/bin/env python

import json
import pickle
import numpy as np

from gryffin import Gryffin

from gemini import GeminiOpt as Gemini
from gemini.plotter import PlotterOpt

import olympus
from olympus.surfaces import Surface
#===============================================================================


#===============================================================================`

'''
Run Gryffin with "external" predictive model.

Predictive model training takes place OUTSIDE of the Gryffin source code,
and a callable predictive model object must then be passed to the recommend
method of gryffin.py

The callable makes predictions about the objective function given parameter
proposals.
This callable must have the following methods and attributes to be compatible with
use in Gryffin

This example uses 2d trigonometric functions for the surfaces

'''

# define params ----------------------------------------------------------------
PLOT         = True
BUDGET_EXP   = 30
BUDGET_CHEAP = 1000
RATIO        = 10
CONFIG_FILE  = 'config.json'
RANDOM_SEED  = 100700
TYPE         = 'gemini'


# helper function
def normalize(x, minmax=None):
	if not type(minmax)==np.ndarray:
		z = (x - np.amin(x))/(np.amax(x)-np.amin(x))
	else:
		z = (x - np.amin(minmax))/(np.amax(minmax)-np.amin(minmax))
	return z

# initialize plotter
if PLOT:
	plotter = PlotterOpt()

# initialize Gryffin
with open(CONFIG_FILE, 'r') as content:
	CONFIG_DICT = json.load(content)
CONFIG_DICT['general']['random_seed'] = RANDOM_SEED

gryffin = Gryffin(config_dict=CONFIG_DICT)

# initalize Gemini
gemini = Gemini()

# define the surfaces ----------------------------------------------------------
def cheap_func(X, Y):
	return np.sin(X+Y)

def exp_func(X, Y):
	return np.cos(X+Y)

# generate the data
RESOLUTION = 50
x = np.linspace(-3, 3, RESOLUTION)
y = np.linspace(-3, 3, RESOLUTION)

X, Y = np.meshgrid(x, y)
cheap_Z = cheap_func(X, Y)
exp_Z   = exp_func(X, Y)

features = np.dstack([X, Y]).reshape(-1, 2)
cheap_z = exp_Z.reshape((x.shape[0], y.shape[0]))
exp_z = exp_Z.reshape((x.shape[0], y.shape[0]))


# START THE CAMPAIGN -----------------------------------------------------------
iteration = 0
observations_exp = []
observations_cheap = []

while len(observations_exp) < BUDGET_EXP:

	# check to see if we have sufficent observations to commence training
	if len(observations_exp) >= 2 and len(observations_cheap) >= 2:
		print('@@\tTRAINING PREDICTIVE MODEL\n')

		training_set = gryffin.construct_training_set(observations_exp, observations_cheap)

		gemini.train(training_set['train_features'], training_set['train_targets'],
                     training_set['proxy_train_features'], training_set['proxy_train_targets'],
                     num_folds=1, user_hyperparams={'max_epochs': 10000})

	#-----------------------------------------------------------------------
	# EXPENSIVE EXPERIMENT
	#-----------------------------------------------------------------------
	print('@@\tEXPENSIVE EXPERIMENT\n')
	# get a new sample
	samples_exp  = gryffin.recommend(observations=observations_exp,
										 predictive_model=gemini)
	# get measurements for samples
	new_observations  = []
	for sample in samples_exp:
		param         = np.array([sample['param_0'][0], sample['param_1'][0]])
		grid = np.meshgrid(param[0], param[1])
		measurement   = exp_func(grid[0], grid[1])
		sample['obj'] = measurement[0][0]
		new_observations.append(sample)
	# add measurements to cache
	observations_exp.extend(new_observations)

	#-----------------------------------------------------------------------
	# CHEAP EXPERIMENT
	#-----------------------------------------------------------------------
	print('@@\tCHEAP EXPERIMENT\n')
	# add same observations as the proxy observations
	samples_cheap = np.random.uniform(-3, 3, size=(2*RATIO, 2))
	new_observations = []
	for sample in samples_cheap:
		grid = np.meshgrid(sample[0], sample[1])
		measurement = cheap_func(grid[0], grid[1])
		new_observations.append({'param_0': np.array([sample[0]]),
								 'param_1': np.array([sample[1]]),
								 'obj': measurement[0][0]})
	# add measurements to cache
	observations_cheap.extend(new_observations)

	# plotting stuff------------------------------------------------------------
	if PLOT:
		# unpack the observations
		params_exp = np.array([[o['param_0'][0], o['param_1'][0]] for o in observations_exp])
		objs_exp   = np.array([o['obj'] for o in observations_exp]).reshape(-1, 1)
		params_cheap = np.array([[o['param_0'][0], o['param_1'][0]] for o in observations_cheap])
		objs_cheap   = np.array([o['obj'] for o in observations_cheap]).reshape(-1, 1)

		if iteration >=1:
			inv_vol = gryffin.bayesian_network.inverse_volume
			kernel = gryffin.bayesian_network.kernel_contribution
			vals_0g = []
			vals_1g = []

			for element in features:
				num, inv_den = kernel(element)
				element = element.reshape(1, len(element))
				g, _ = gemini.predict(element)
				vals_0g.append((num + gryffin.acquisition.sampling_param_values[0] + inv_vol*gemini.get_pearson_coeff()* g) * (inv_den / (inv_vol * inv_den + 1)))
				vals_1g.append((num + gryffin.acquisition.sampling_param_values[1] + inv_vol*gemini.get_pearson_coeff()* g) * (inv_den / (inv_vol * inv_den + 1)))
			vals_0g = np.squeeze(np.array(vals_0g))
			vals_1g = np.squeeze(np.array(vals_1g))

		# normalize everything
		obs_cheap = normalize(objs_cheap, minmax=cheap_z)
		cheap_z = normalize(cheap_z)
		obs_exp   = normalize(objs_exp, minmax=exp_z)
		exp_z = normalize(exp_z)

		vals_0g_mesh=None
		vals_1g_mesh=None
		if iteration >= 1:
			vals_0g = normalize(vals_0g)
			vals_0g_mesh = vals_0g.reshape((x.shape[0], y.shape[0]))
			vals_1g = normalize(vals_1g)
			vals_1g_mesh = vals_1g.reshape((x.shape[0], y.shape[0]))


		plotter.plot_monitor(iteration, X, Y, cheap_Z, exp_Z,
							 params_cheap, params_exp,
							 objs_cheap, objs_exp,
							 RATIO,
		 					 vals_0g_mesh=vals_0g_mesh,
							 vals_1g_mesh=vals_1g_mesh,
							 )

	iteration+=1
