#!/usr/bin/env python

import pickle
import itertools
import random
import numpy as np

#==================================================================

'''
	Goal: minimize the bandgap
'''

#==================================================================

organics = [
		'acetamidinium', 'ammonium', 'azetidinium', 'butylammonium', 'dimethylammonium',
        'ethylammonium', 'formamidinium', 'guanidinium', 'hydrazinium', 'hydroxylammonium',
        'imidazolium', 'isopropylammonium', 'methylammonium', 'propylammonium', 'tetramethylammonium', 'trimethylammonium',
]
cations  = ['Ge', 'Sn', 'Pb']
anions   = ['F', 'Cl', 'Br', 'I']

#==================================================================

class Evaluator(object):

	func_name = None
	func_doc  = None

	def __init__(self, lookup_file = 'lookup_table.pkl',
				 type = 'slow', log_name = None,
				 max_iter = 200, separate = False):

		self.lookup_file = lookup_file
		with open(self.lookup_file, 'rb') as content:
			self.lookup_table = pickle.load(content)

		self.keys = {'slow': 'bandgap_hse06', 'fast': 'bandgap_gga'}
		self.type = type
		self.vectors  = []
		self.ranks    = []
		self.bandgaps = []
		self.counter  = 0
		self.max_iter = max_iter
		self.log_name = log_name

		self.eval_dict = {}

		# create ranks from lookup table
		lowest_bandgaps = []
		for organic in organics:
			organic = organic.capitalize()
			for cation in cations:
				for anion in anions:
					bandgaps = self.lookup_table[organic][cation][anion][self.keys[self.type]]
					lowest_bandgap = np.amin(bandgaps)
					self.lookup_table[organic][cation][anion]['lowest_bandgap'] = lowest_bandgap
					lowest_bandgaps.append(lowest_bandgap)

		lowest_bandgaps = np.array(lowest_bandgaps)
		sort_indices    = np.argsort(lowest_bandgaps)
		sorted_bandgaps = lowest_bandgaps[sort_indices]

		for organic in organics:
			organic = organic.capitalize()
			for cation in cations:
				for anion in anions:
					bandgap = self.lookup_table[organic][cation][anion]['lowest_bandgap']
					index   = np.where(bandgap <= sorted_bandgaps)[0][0] + 1
					self.lookup_table[organic][cation][anion]['rank'] = index


		with open('CatDetails/cat_details_organic.pkl', 'rb') as content:
			self.organics = pickle.load(content)
		with open('CatDetails/cat_details_anion.pkl', 'rb') as content:
			self.anions = pickle.load(content)
		with open('CatDetails/cat_details_cation.pkl', 'rb') as content:
			self.cations = pickle.load(content)



	def generate_dataset(self):
		''' generate all the potenital combinations
		'''
		idxs = [
			[c['name'] for c in self.cations],
			[a['name'] for a in self.anions],
			[o['name'] for o in self.organics],
		]
		return list(itertools.product(*idxs))


	def select_random(self, options, num_samples):
		''' select randomly num samples perovskites and remove then from
		options
		'''
		samples = []
		samples = random.sample(options, num_samples)
		for sample in samples:
			options.remove(sample)
		return samples, options


	def __call__(self, vector):

		print('VECTOR (%d)\t' % self.counter, vector)


		organic_name = vector['organic'][0].capitalize()
		organic_name_lower = vector['organic'][0]
		anion_name   = vector['anion'][0]
		cation_name  = vector['cation'][0]

		# get descriptor vector
		organic_desc = list(filter(lambda organic: organic['name'] == organic_name_lower, self.organics))[0]['descriptors']

		anions_desc = list(filter(lambda anion: anion['name'] == anion_name, self.anions))[0]['descriptors']

		cations_desc = list(filter(lambda cation: cation['name'] == cation_name, self.cations))[0]['descriptors']

		descriptor = np.concatenate((organic_desc, anions_desc, cations_desc), axis=0)


#		organic_name  = self.organics[organic_index]['name'].capitalize()
#		anion_name    = self.anions[anion_index]['name']
#		cation_name   = self.cations[cation_index]['name']

		vector_string = '%s_%s_%s' % (organic_name, cation_name, anion_name)
		if vector_string in self.eval_dict:
			return self.eval_dict[vector_string]


		# evaluate
		bandgap = self.lookup_table[organic_name][cation_name][anion_name]['lowest_bandgap']
		rank    = self.lookup_table[organic_name][cation_name][anion_name]['rank']
		self.eval_dict[vector_string] = bandgap

		# log results in memory
		self.vectors.append(vector)
		if len(self.bandgaps) == 0:
			self.bandgaps.append(bandgap)
			self.ranks.append(rank)
		else:
			if bandgap < self.bandgaps[-1]:
				self.bandgaps.append(bandgap)
				self.ranks.append(rank)
			else:
				self.bandgaps.append(self.bandgaps[-1])
				self.ranks.append(self.ranks[-1])

		# log results on disk
		if not self.log_name is None:
			report = '%d\t%.5e\t%d\t%s\t%s\t%s\n' % (self.counter, bandgap, rank, organic_name, anion_name, cation_name)
			log_file = open(self.log_name, 'a')
			log_file.write(report)
			log_file.close()

		self.counter += 1

		if self.counter == self.max_iter:
			print('reached max_iter', self.counter, self.max_iter)
			quit()
		return bandgap

#==================================================================

if __name__ == '__main__':

	# vector type: {'organic': ['propylammonium'], 'cation': ['Pb'], 'anion': ['Br']}
	vector =  {'organic': ['propylammonium'], 'cation': ['Pb'], 'anion': ['Br']}

	evaluator_exp = Evaluator(type='slow')
	# evaluate the HSE06 data
	hse06_bandgap, descriptor = evaluator_exp(vector)
	print('HSE06 bandgap (slow) : ', hse06_bandgap)
	print('DESCRIPTOR : ', descriptor)
	print(descriptor.shape)

	evaluator_cheap = Evaluator(type='fast')
	# evaluate the GGA data
	gga_bandgap = evaluator_cheap(vector)
	print('GGA bandgap (slow) : ', gga_bandgap)
