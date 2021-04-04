import glob

from .dataset import Dataset
from gemini import __datasets__

#==============================================================================

def get_avail_datasets():
    dataset_names = glob.glob(f'{__datasets__}/dataset_*/')
    dataset_names = [dataset.split('/')[-2] for dataset in dataset_names]
    return dataset_names
