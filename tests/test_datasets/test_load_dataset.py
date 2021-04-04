#!/usr/bin/env python

import pytest
import glob
import numpy as np

import gemini
from gemini import __datasets__
from gemini.datasets import Dataset

#===============================================================================

DATASET_NAMES = ['freesolv', 'photobleaching', 'cat_oer_1_4', 'perovskite']

def check_dataset_obj():
    #dataset_names = gemini.datasets.get_avail_datasets()
    for name in DATSET_NAMES:
        dataset = Dataset(name = name)
        assert(type(dataset) == gemini.datasets.dataset.Dataset)



if __name__ == '__main__':
    assert(type(gemini.datasets.get_avail_datasets()) == list)
    check_dataset_obj()
