import os
import sys
import time
from collections import OrderedDict
import inspect
import pickle
from tqdm import tqdm

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Chemo-informatics
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, PandasTools
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
print('RDKit:{}'.format(rdkit.__version__))
# ML/DL
import sklearn


#=====================================

class RdkitDescriptors():
    def __init__(self, PATH, target_col_fast, target_col_slow, smiles_col):

#        self.df = pickle.load(open(PATH, 'rb'))
        self.df = pd.read_csv(PATH, delimiter = ',')
#        self.df = pd.read_excel(PATH)
        print(self.df.head())
        self.target_col_fast = target_col_fast
        self.target_col_slow = target_col_slow
        self.smiles_col = smiles_col
        data = {'smiles': self.df[smiles_col], 'target_fast': self.df[target_col_fast], 'target_slow':self.df[target_col_slow]}
        self.df = pd.DataFrame(data)
        print(self.df.head())
        self.df['mol'] = self.df.apply(lambda row: Chem.MolFromSmiles(row['smiles']), axis = 1)
        print(self.df.head())


    def basic_rdkit(self):
        """
        Returns a dataframe with the ~200 cheap conformer independent
        descriptors availbale from RDKit
        """
        props_to_use = list()
        calc_props = OrderedDict(inspect.getmembers(Descriptors, inspect.isfunction))
        print(calc_props.keys())
        for key in list(calc_props.keys()):
            if key.startswith('_'):
                del calc_props[key]
        print('Found {} molecular descriptors in RDKIT'.format(len(calc_props)))
        for key,val in calc_props.items():
            self.df[key] = self.df['mol'].apply(val)
            if not True in np.isnan(self.df[key].tolist()):
                props_to_use.append(key)
        print("DF shape: ", self.df.shape)
        print(self.df.head())

        return self.df



    def rdkit_fingerprints(self):
        """
        Returns a dataframe including the RDKit molecular fingerprints
        """
        self.df['rdkitfp'] = self.df.apply(lambda row: Chem.RDKFingerprint(row['mol']), axis = 1)
#        x_fp = self.df['rdkitfp'].values
#        print(x_fp.shape)
        return self.df


    def pickle_df(self):
        """
        Pickle the pandas Dataframe containing the rdkit features
        """
        file = 'rdkit_feat.pickle'
        pickle_out = open(file, 'wb')
        pickle.dump(self.df, pickle_out)
        pickle_out.close()

        return None


#==========================================================================
# debugging
rdkitdesc = RdkitDescriptors('SAMPL.csv', 'calc', 'expt', 'smiles')
df = rdkitdesc.basic_rdkit()
print(df.head())
print('='*75)
fast_targets = df['target_fast'].values
slow_targets = df['target_slow'].values
features = df.values[:, 4:]
columns = df.columns
columns = columns[4:]
print(columns)
print(features.shape, fast_targets.shape, slow_targets.shape)

print(features)
# ============================
# select most important features with gradient boosting regressor
from sklearn.model_selection import  train_test_split

train_index, test_index = train_test_split(np.arange(features.shape[0]), test_size = 0.20, shuffle = True)
train_features, train_targets = features[train_index], slow_targets[train_index]
test_features, test_targets   = features[test_index], slow_targets[test_index]

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

kf = KFold(n_splits=10)
results=[]
for train, test in tqdm(kf.split(features),total=10):
  train_features, test_features = features[train], features[test]
  train_targets, test_targets = slow_targets[train], slow_targets[test]
  model =  GradientBoostingRegressor()
  model.fit(train_features, train_targets.ravel())
  pred_targets = model.predict(test_features)
  results.append(sklearn.metrics.r2_score(test_targets, pred_targets))
print(np.mean(results), np.std(results))

feature_importances = model.feature_importances_

important_ix = feature_importances.argsort()[-10:][::-1]
print(important_ix)
print(feature_importances[important_ix])
print(columns[important_ix])

# get the good dataset
features = features[:, important_ix]

data = {'fast_features': features, 'slow_features': features, 'fast_targets': fast_targets, 'slow_targets': slow_targets}
print(data)
print(data['fast_features'].shape, data['fast_targets'].shape, data['slow_targets'].shape)

with open('dataset.pkl', 'wb') as content:
    pickle.dump(data, content)
