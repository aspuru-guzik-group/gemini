#!/usr/bin/env python

#==========================================================================

import os, sys
import numpy as np
import time
import pickle

from evaluator import Evaluator

from gryffin import Gryffin

#==========================================================================

BUDGET_EXP   = 192
BUDGET_CHEAP = 192
RATIO        = 10
SEED		 = 100700
CONFIG_FILE_EXP   = 'config_exp.json'
CONFIG_FILE_CHEAP = 'config_cheap.json'

LOG_NAME_EXP    = f'logfile_exp_{RATIO}_{SEED}.dat'
LOG_NAME_CHEAP  = f'logfile_chep_{RATIO}_{SEED}.dat'

#==========================================================================

if __name__ == '__main__':

	for filename in [LOG_NAME_CHEAP, LOG_NAME_EXP]:
		if os.path.isfile(filename):
			print('@@\tRemoving old log file...')
			os.remove(filename)

	# set up the evaluator - lookup table
	evaluator_exp   = Evaluator(log_name = LOG_NAME_EXP, type = 'slow')
	evaluator_cheap = Evaluator(log_name = LOG_NAME_CHEAP, type = 'fast')

	# instantiate gryffin
	gryffin   = Gryffin(config_file = CONFIG_FILE_EXP)

	# START THE EXPERIMENT
	iteration = 0
	num_eval_cheap = 0
	num_eval_exp   = 0
	observations_exp = []
	observations_cheap = []

	while num_eval_exp < BUDGET_EXP:
		#-----------------------------------------------------------------------
		# EXPENSIVE EXPERIMENT
		#-----------------------------------------------------------------------

		print('@@\tEXPENSIVE EXPERIMENT\n')

		samples_exp = gryffin.recommend(observations=observations_exp,
											proxy_observations=observations_cheap)
		# get measurements on the expensive samples
		new_observations_exp = []
		for sample in samples_exp:
			measurement = evaluator_exp(sample)
			sample['bandgap'] = measurement
			new_observations_exp.append(sample)
		observations_exp.extend(new_observations_exp)
		num_eval_exp += gryffin.config.get('sampling_strategies')


		#-----------------------------------------------------------------------
		# CHEAP EXPERIMENT
		#-----------------------------------------------------------------------

		print('@@\tCHEAP EXPERIMENT\n')

		# random sampling on the cheap surface


		if num_eval_cheap <= BUDGET_CHEAP-(gryffin.config.get('sampling_strategies')*RATIO):
			samples_cheap = []
			for i in range(RATIO):
				sample_cheap = gryffin_cheap.recommend(observations = [observations_exp, observations_cheap])
				samples_cheap.extend(sample_cheap)

			# get measurements on the cheap points
			new_observations_cheap = []
			for sample in samples_cheap:
				measurement = evaluator_cheap(sample)
				sample['bandgap'] = measurement
				new_observations_cheap.append(sample)
			observations_cheap.extend(new_observations_cheap)
			num_eval_cheap += RATIO*gryffin.config.get('sampling_strategies')


		print('NUM OBSERVATIONS EXP :', len(observations_exp))
		print('NUM OBSERVATIONS CHEAP : ', len(observations_cheap))

		observations = {'observations_exp': observations_exp, 'observations_cheap': observations_cheap}
		pickle.dump(observations, open(f'opt_runs/run_{ratio}_{seed}.pkl', 'wb'))
		# clean up the svaed weights for this run
