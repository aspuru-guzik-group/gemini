#/usr/bin/env python

import os, sys
import time
import numpy as np
import tensorflow as tf

#==============================================================================-

from gemini.base import Base
from gemini.metrics import Metrics
from gemini.utils import Logger, get_args

#===============================================================================

class GeminiOpt(Base, Logger):
    '''
    Wrapper for Gemini for use in closed-loop optimization setting
    '''
    def __init__(self,
                **kwargs
        ):
        super().__init__(**get_args(**locals()))
        Logger.__init__(self, 'GeminiOpt', verbosity=0)
        self.is_opt = True
        self.is_internal = False
        self.name = 'gemini'
        
    def estimate_pearson_coeff(self,
                              features,
                              targets):
        ''' estimate pearsons coefficient
        '''
        if not self.cv_models:
            self.load_full_model()
            mean_pred, _ = self.predict(features)
        else:
            self.load_cv_models()
            mean_pred, _ = self.predict_cv(features)

        self.pearson_coeff = Metrics.pearson(targets, mean_pred)
        self.log(f'Pearson coefficient computed : rho={round(self.pearson_coeff, 2)}', 'INFO')


    def get_pearson_coeff(self):
        return self.pearson_coeff


    def get_num_folds(self):
        return self.num_folds



if __name__ == '__main__':
	# # DEBUG:

	opt = GeminiOpt(model_type='gemini', model_path='./.tmp_models')
