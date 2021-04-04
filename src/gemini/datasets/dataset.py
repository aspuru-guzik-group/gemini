#!/usr/bin/env python

import os
import glob
import pickle
import numpy as np

#===============================================================================

from gemini.utils import Logger
from gemini import __datasets__

#===============================================================================

class Dataset(Logger):
    ''' 'real-life' dataset to be used with Gemini
        Datasets are stored in the __datasets__/ directory with the directory name
        dataset_name. Each dataset directory should contain a parsed version of the
        dataset as a dictionary in the file dataset.pkl. This dictionary should have
        the keys 'train_fast_features', 'train_fast_targets', 'train_slow_features',
        'train_slow_targets'. The correspnding values should be numpy arrays containing
        the raw data
    '''
    def __init__(self, name):
        '''
        Args:
            name (str): the name of the dataset
        '''
        self.name = name
        Logger.__init__(self, 'Dataset', verbosity=0)
        #--------------------------
        # Load dataset
        #--------------------------
        if name != None:
            self.data, self.config, self.description = self._load_dataset(self.name)

        if len(self.data['slow_targets'].shape) == 1:
            self.data['slow_targets'] = self.data['slow_targets'].reshape(-1, 1)
        if len(self.data['fast_targets'].shape) == 1:
            self.data['fast_targets'] = self.data['fast_targets'].reshape(-1, 1)

        self.slow_features_shape = self.data['slow_features'].shape[1]
        self.slow_targets_shape  = self.data['slow_targets'].shape[1]
        self.fast_features_shape = self.data['fast_features'].shape[1]
        self.fast_targets_shape  = self.data['fast_targets'].shape[1]

        self.total_fast_obs = self.data['slow_features'].shape[0]
        self.total_slow_obs = self.data['fast_features'].shape[0]


    def load_full(self, alpha):
        ''' Load all of the data associated with the training of a particular
        Gemini dataset (all num slows and all folds)

        Args:
            r (int): how many times more fast observations we have than slow ones

        Returns:
            data (list): list of dictionaries containing all of the specific
                         subdatasets
        '''
        data = []
        num_slow_arr = self.get_num_slow_arr(r = r)
        for num_slow in num_slow_arr:
            data.append({'num_slow': num_slow, 'data': self.load_specific(num_slow, r)})
        return data


    def load_specific(self, num_slow, r = 5):
        ''' Load the training and validation data for a specific number of slow
        evalutations (all folds)

        Args:
            num_slow (int): indicates the number of slow evaluations in the specific
                            subdataset to be loaded
            r (int): how many times more fast observations we have than slow ones

        Returns:
            specific_data (dict): dictionary containing the training and validation data
                                  for each evaluator for a specific number of slow
                                  evaluations
        '''
        indices = self._load_indices(self.name, r)
        try:
            specific_indices = list((filter(lambda d: d['num_slow'] == num_slow, indices)))[0]
        except NotImplementedError:
            self.log('Indices for this number of expensive evaluations are not stored', 'FATAL')

        num_samples = len(specific_indices['cv_folds']['train_slow_indices'])
        train_slow_indices = np.array(specific_indices['cv_folds']['train_slow_indices'])
        test_slow_indices  = np.array(specific_indices['cv_folds']['test_slow_indices'])
        train_fast_indices = np.array(specific_indices['cv_folds']['train_fast_indices'])
        test_fast_indices  = np.array(specific_indices['cv_folds']['test_fast_indices'])

        specific_data = self._make_train_valid_set(train_slow_indices, test_slow_indices,
                                                   train_fast_indices, test_fast_indices,
                                                   num_samples)
        return specific_data


    def load_emulator(self, eval):
        ''' Load in the emulator data, if it exists
        '''
        emulator_indices = self.load_emulator_indices(eval)

        data = self.make_emulator_train_test_set(emulator_indices[eval]['train_indices'],
                                            emulator_indices[eval]['test_indices'],
                                            eval = eval)
        return data


    def _make_train_valid_set(self, train_slow_indices, test_slow_indices,
                              train_fast_indices, test_fast_indices, num_samples):
        ''' Generate training and validation data for a specific set of indices

        Args:
            num_samples (int): The number of random samples to take from the dataset
        '''

        train_fast_features = np.empty((num_samples, train_fast_indices.shape[1], self.fast_features_shape))
        train_fast_targets  = np.empty((num_samples, train_fast_indices.shape[1], self.fast_targets_shape))
        test_fast_features  = np.empty((num_samples, test_fast_indices.shape[1],  self.fast_features_shape))
        test_fast_targets   = np.empty((num_samples, test_fast_indices.shape[1],  self.fast_targets_shape))

        train_slow_features = np.empty((num_samples, train_slow_indices.shape[1], self.slow_features_shape))
        train_slow_targets  = np.empty((num_samples, train_slow_indices.shape[1], self.slow_targets_shape))
        test_slow_features  = np.empty((num_samples, test_slow_indices.shape[1],  self.slow_features_shape))
        test_slow_targets   = np.empty((num_samples, test_slow_indices.shape[1],  self.slow_targets_shape))


        for _, sample in enumerate(train_fast_indices):
            train_fast_features[_] = self.data['fast_features'][sample]
            train_fast_targets[_]  = self.data['fast_targets'][sample]

        for _, sample in enumerate(test_fast_indices):
            test_fast_features[_] = self.data['fast_features'][sample]
            test_fast_targets[_]  = self.data['fast_targets'][sample]

        for _, sample in enumerate(train_slow_indices):
            train_slow_features[_] = self.data['slow_features'][sample]
            train_slow_targets[_]  = self.data['slow_targets'][sample]

        for _, sample in enumerate(test_slow_indices):
            test_slow_features[_] = self.data['slow_features'][sample]
            test_slow_targets[_]  = self.data['slow_targets'][sample]

        return {'train_fast_features': train_fast_features, 'train_fast_targets': train_fast_targets,
                'test_fast_features': test_fast_features, 'test_fast_targets': test_fast_targets,
                'train_slow_features': train_slow_features, 'train_slow_targets': train_slow_targets,
                'test_slow_features': test_slow_features, 'test_slow_targets': test_slow_targets}


    def make_emulator_train_test_set(self, train_indices, test_indices, eval):
        ''' Construct the datasets for an emulator
        '''
        train_features = self.data[f'{eval}_features'][train_indices]
        train_targets  = self.data[f'{eval}_targets'][train_indices]
        test_features  = self.data[f'{eval}_features'][test_indices]
        test_targets   = self.data[f'{eval}_targets'][test_indices]

        return {'train_features': train_features, 'train_targets': train_targets,
                'test_features': test_features, 'test_targets': test_targets}

    def _load_dataset(self, name):
        ''' Load the entire datasets dictionary for this dataset. Must be a file
        named dataset.pkl stored in the directory __datasets__/name where name is
        the name of the dataset
        '''
        # TODO: implement a description and configuration (i.e. description of features
        # and so-on ) for each dataset
        # load description
        description = None
        # load info
        config = None
        # load datasets
        csv_file = f'{__datasets__}/dataset_{name}/dataset.pkl'
        try:
            data = pickle.load(open(csv_file, 'rb'))
        except FileNotFoundError:
            print(f'Could not find the dataset named {name}')

        return data, config, description


    def generate_indices(self, data, num_slow_arr, r, num_samples,
                          seed = 100700, to_disk = True):
        ''' Generate new indces for a particular dataset. The file containing the
        indices will be saved as __datasets__/dataset_name/indices_r.pkl if the
        to_disk argument is set to True. These indices
        can then be loaded by the _load_dataset method of this object.

        # NOTE/WARNING: For now, calling this method will overwrite previous indices files of
        the same name!!

        Args:
            data (dict): dictionary containing the fast and slow data

            num_slow_arr (array-like): list of integers corresponding to # of slow evaluations
            r (int): how many times more fast evalautions to use than slow ones
            num_samples (int): the number of random samples draw per num_slow
            seed (int): random seed for numpy
            to_disk (bool): whether or not to pickle the indices

        Return:
            indices (list): list of dictionaries containing indices for all num_slows
                            and all samples
        '''
        # if os.path.exists(f'{__datasets__}/{self.name}/indices_{r}.pkl'):
        #     print('There already exist indices for this dataset with this value of r!')


        num_obs_fast = data['fast_features'].shape[0]
        num_obs_slow = data['slow_features'].shape[0]


        num_fast_arr     = np.array(num_slow_arr) * r
        num_fast_arr = [i if i <= num_obs_fast else num_obs_fast-2 for i in num_fast_arr]

        indices = []

        slow_indices, fast_indices = np.arange(num_obs_slow), np.arange(num_obs_fast)

        for ix, num_slow in enumerate(num_slow_arr):
            num_fast = num_fast_arr[ix]

            np.random.shuffle(slow_indices)
            np.random.shuffle(fast_indices)

            folds = {'train_fast_indices': [], 'test_fast_indices': [],
                     'train_slow_indices': [],  'test_slow_indices': []}

            for fold in range(num_samples):

                train_slow_indices, test_slow_indices = slow_indices[:num_slow], slow_indices[num_slow:]

                train_fast_indices, test_fast_indices = fast_indices[:num_fast], fast_indices[num_fast:]

                folds['train_fast_indices'].append(train_fast_indices)
                folds['test_fast_indices'].append(test_fast_indices)
                folds['train_slow_indices'].append(train_slow_indices)
                folds['test_slow_indices'].append(test_slow_indices)

                slow_indices = np.roll(slow_indices, num_slow)
                fast_indices = np.roll(fast_indices, num_fast)

            indices.append({'num_slow': num_slow, 'num_fast': num_fast,
                             'cv_folds': folds})

        if to_disk:
            pickle.dump(indices, open(f'{__datasets__}/dataset_{self.name}/indices_{r}.pkl', 'wb'))

        return indices


    def generate_emulator_indices(self, data, frac_test, seed = 100700, to_disk = True):
        ''' Generate indices for training the BNN emulator on the datasets which are used
        for optimizations

        Args:
            data (dict): dictionary containing the fast and slow data
            frac_test (float): the fraction of the dataset to be used for testing/vaidation
            seed (int): random seed for numpy
            to_disk (bool): whether or not to pickle the indices

        Return:
            indices (list): dictionary containing indices
        '''
        emulator_indices = {}
        for eval in ['fast', 'slow']:
            num_obs = data[f'{eval}_features'].shape[0]
            num_train = int((1-frac_test) * num_obs)
            indices = np.arange(num_obs)
            np.random.shuffle(indices)
            train_indices = indices[:num_train]
            test_indices  = indices[num_train:]

            emulator_indices[eval] = {'train_indices': train_indices, 'test_indices': test_indices}

            if to_disk:
                pickle.dump(emulator_indices, open(f'{__datasets__}/dataset_{self.name}/indices_emulator_{eval}.pkl', 'wb'))

        return emulator_indices


    def _load_indices(self, name, r):
        ''' Load the stored indices from disk if they exist
        '''
        indices_file = f'{__datasets__}/dataset_{name}/indices_{r}.pkl'
        try:
            indices = pickle.load(open(indices_file, 'rb'))
        except FileNotFoundError:
            print(f'Could not find indcies with r={r} for dataset named {name}')

        return indices

    def load_emulator_indices(self, eval):
        ''' Load the stored indices from disk for a particular emulator
        '''
        indices_file = f'{__datasets__}/dataset_{self.name}/indices_emulator_{eval}.pkl'
        try:
            indices = pickle.load(open(indices_file, 'rb'))
        except FileNotFoundError:
            print(f'Could not find indcies for {eval} emulator for dataset named {self.name}')

        return indices

    def get_num_slow_arr(self, alpha):
        ''' Returns a list of the amount of slow evalautions in each of the
            sub-datasets for this dataset object
        '''
        indices = self._load_indices(self.name, alpha)
        num_slow_arr = [i['num_slow'] for i in indices]
        return num_slow_arr

    def get_num_fast_arr(self, alpha):
        ''' Returns a list of the amount of fast evalautions in each of the
            sub-datasets for this dataset object (this should always be
            # fast = # slow * alpha)
        '''
        indices = self._load_indices(self.name, alpha)
        num_fast_arr = [i['num_fast'] for i in indices]
        return num_fast_arr
