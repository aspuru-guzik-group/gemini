#!/usr/bin/env python 

import os
import copy
import json
import pickle 
import numpy as np 

#=================================================================

class CategoryWriter(object):

	param_names = ['anion', 'cation', 'organic']

	desc_names  = {
			'anion':   ['electron_affinity', 'ionization_energy', 'mass', 'electronegativity'],
			'cation':  ['electron_affinity', 'ionization_energy', 'mass', 'electronegativity'],
			'organic': ['homo', 'lumo', 'dipole', 'atomization', 'r_gyr', 'total_mass']
		}	  


	def __init__(self):
		self.anions   = json.loads(open('anions.json', 'r').read())
		self.cations  = json.loads(open('cations.json', 'r').read())	
		self.organics = json.loads(open('organics.json', 'r').read())
		self.opts     = {'anion': self.anions, 'cation': self.cations, 'organic': self.organics}


	def write_categories(self, home_dir, with_descriptors = True):
		for param_name in self.param_names:
			opt_list = []
			for opt_name, opt_desc_dict in self.opts[param_name].items():
				opt_dict = {'name': opt_name}
				if with_descriptors:
					opt_dict['descriptors'] = np.array([float(opt_desc_dict[desc_name]) for desc_name in self.desc_names[param_name]])
				opt_list.append(copy.deepcopy(opt_dict))

			# create cat details dir if necessary
			dir_name = '%s/CatDetails' % home_dir
			if not os.path.isdir(dir_name): os.mkdir(dir_name)
		
			cat_details_file = '%s/cat_details_%s.pkl' % (dir_name, param_name)
			with open(cat_details_file, 'wb') as content:
				pickle.dump(opt_list, content)
							
#=================================================================

if __name__ == '__main__':

	cat_writer = CatWriter()
	cat_writer.write_cats('./')


