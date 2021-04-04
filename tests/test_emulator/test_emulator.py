# #!/usr/bin/env python
#
# import pytest
#
# import numpy as np
#
# from gemini import __datasets__
# from gemini.datasets import Dataset
# from gemini.emulators import Emulator
#
# #===============================================================================
#
# def train_emulator(dataset_name = 'photobleaching'):
#
#     dataset = Dataset(name = dataset_name)
#
#     emulator_indices = dataset.generate_emulator_indices(dataset.data, frac_test=0.2)
#
#     fast_data = dataset.load_emulator(eval = 'fast')
#     slow_data = dataset.load_emulator(eval = 'slow')
#
#     assert(type(fast_data) == dict)
#     assert(type(slow_data) == dict)
#
#     hyparams = {'max_epochs': 1*10**3}
#
#     emulator = Emulator(name = dataset_name, eval = 'fast')
#     emulator.create_model(hyperparams = hyparams)
#     emulator.build_model()
#     emulator.train_()
#
#
# def load_emulator(dataset_name = 'photobleaching'):
#     pass
#
#
# def predict_emulator(dataset_name):
#     pass
#
#
# if __name__ == '__main__':
#     test_emulator()
