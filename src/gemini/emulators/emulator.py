#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from gemini import  __datasets__
from gemini.utils import get_args
from gemini.datasets import Dataset
from gemini.data_transformer import DataTransformer
from gemini.models import BayesianNeuralNetwork


#===============================================================================

class Emulator(Logger):

    def __init__(self,
                name = 'photobleaching',
                eval = 'slow',
                hyparams = None,
                **kwargs,):
        Logger.__init__(self, )
        self.name = name
        self.eval = eval
        self.model_path = f'{__datasets__}/dataset_{self.name}/emulator_{self.eval}'
        self.dataset = self.load_data()

    def load_data(self):
        dataset_object = Dataset(self.name)
        self.features_shape = getattr(dataset_object, f'{self.eval}_features_shape')
        self.targets_shape  = getattr(dataset_object, f'{self.eval}_targets_shape')
        data = dataset_object.load_emulator(eval = self.eval)
        return data

    def create_model(self, hyperparams={}):
        self.graph = tf.Graph()
        self.model = BayesianNeuralNetwork(graph = self.graph,
                                           features_shape = self.features_shape,
                                           targets_shape = self.targets_shape,
                                           **hyperparams)

    def build_model(self):
        with self.graph.as_default():
            self.model.build_graph()


    def load_model(self, hyperparams={}):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = BayesianNeuralNetwork(graph = self.graph,
                                               features_shape = self.features_shape,
                                               targets_shape = self.targets_shape,
                                               **hyperparams)
            self.model.build_graph()
            if self.model.restore(f'{self.model_path}/model.ckpt'):
                return self.model
            else:
                return None

    def train_(self):
        stats_dict = DataTransformer().dataset_stats(self.dataset['train_features'],
                                                     self.dataset['train_targets'],
                                                     None,
                                                     None)

        self.model.train(self.dataset['train_features'], self.dataset['train_targets'],
                         self.dataset['test_features'], self.dataset['test_targets'],
                         stats_dict, self.model_path, plot=False)


    def predict(self, features, evaluator):
        mean_pred, std_pred = self.model.run_prediction(features, evaluator)
        return mean_pred
