#!/usr/bin/env python

import numpy as np


#====================================================================


#====================================================================

# GENERAL PURPOSE SIMPLEX TRANSFORMATION
def cube_to_simpl(cubes):
	'''
	converts and n-cube (used for optimization) to an n+1 simplex (used
	as features for Gemini)
	'''
	features = []
	for cube in cubes:
		cube = (1 - 2 * 1e-6) * np.squeeze(np.array([c for c in cube])) + 1e-6
		simpl = np.zeros(len(cube)+1)
		sums = np.sum(cube / (1 - cube))

		alpha = 4.0
		simpl[-1] = alpha / (alpha + sums)
		for _ in range(len(simpl)-1):
			simpl[_] = (cube[_] / (1 - cube[_])) / (alpha + sums)
		features.append(np.array(simpl))
	return np.array(features)

def identity(cubes):
	'''
	perform identity transformation on a n-cube (used for optimization and prediction)
	'''
	return cubes


def opt_transform(cubes, kind = 'identity'):
	'''
	general purpose method for transforming n-dimensional cube used for optimization
	with Phoenics to geometric object used as features for Gemini prediction
	'''
	if kind == 'identity':
		return identity(cubes)
	elif kind == 'simpl':
		return cube_to_simpl(cubes)
	else:
		return NotImplementedError



if __name__ == '__main__':
	# test n=5
	cubes = [[0.9, 0.05, 0.2, 0.3, 0.5]]
	simpl = cube_to_simpl(cubes)
	print(simpl)
	print(np.sum(simpl))


	simpl = opt_transform(cubes, kind='simpl')
	print(simpl)

	cubes2 = opt_transform(cubes, kind='identity')
	print(cubes2)


	print('='*50)

	# test n=3
	cubes = [[0.1, 0.7, 0.5], [0.8, 0.4, 0.6]]
	simpl = cube_to_simpl(cubes)
	print(simpl)
	print([np.sum(s) for s in simpl])



#===============================================================================
# # CUBE TO SIMPLEX TRANSFORMATION USED IN PHOTOBLEACHING EXAMPLE
# def cube_to_simpl_photobl(mats):
# 	'''
# 	converts 3-cube (used for optimization) to a 4-simplex (used as features for Gemini)
# 	'''
# 	features = []
# 	cube     = (1 - 2 * 1e-6) * np.squeeze(np.array([mat, mat_1, mat_2])) + 1e-6
# 	simpl    = np.zeros(4)
# 	#print('CUBE', cube)
# 	sums     = np.sum(cube / (1 - cube))
#
# 	alpha    = 4.0
# 	simpl[3] = alpha / (alpha + sums)
# 	simpl[0] = (cube[0] / (1 - cube[0])) / (alpha + sums)
# 	simpl[1] = (cube[1] / (1 - cube[1])) / (alpha + sums)
# 	simpl[2] = (cube[2] / (1 - cube[2])) / (alpha + sums)
# 	feature = np.array([simpl])
#
# 	return feature
#
#
# # CUBE TO SIMPLEX TRANSFORMATION USED IN CAT OER TESTS
# #def cube_to_simpl(elem_0, elem_1, elem_2, elem_3, elem_4):
# def cube_to_simpl_cat_oer(elems):
# 	'''
# 	converts 5-cube (used for optimization) to a 6-simplex (used as features for Gemini)
# 	'''
# 	features = []
# 	for elem in elems:
# 		cube     = (1 - 2 * 1e-6) * np.squeeze(np.array([elem[0], elem[1], elem[2], elem[3], elem[4]])) + 1e-6
# 		simpl    = np.zeros(6)
# 		sums     = np.sum(cube / (1 - cube))
#
# 		alpha    = 4.0
# 		simpl[5] = alpha / (alpha + sums)
# 		simpl[0] = (cube[0] / (1 - cube[0])) / (alpha + sums)
# 		simpl[1] = (cube[1] / (1 - cube[1])) / (alpha + sums)
# 		simpl[2] = (cube[2] / (1 - cube[2])) / (alpha + sums)
# 		simpl[3] = (cube[3] / (1 - cube[3])) / (alpha + sums)
# 		simpl[4] = (cube[4] / (1 - cube[4])) / (alpha + sums)
# 		feature = np.array([simpl])
# 		features.append(np.squeeze(feature))
# 	return np.array(features)
