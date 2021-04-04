#!/usr/bin/env python

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

#===============================================================================


#===============================================================================

class Metrics:

    @staticmethod
    def r2(true, pred):
            return r2_score(true, pred)

    @staticmethod
    def rmse(true, pred):
            return np.sqrt(mean_squared_error(true, pred))

    @staticmethod
    def mae(true, pred):
            return mean_absolute_error(true, pred)

    @staticmethod
    def pearson(true, pred):
        if true.shape[-1] == 1:
            true, pred = np.squeeze(true), np.squeeze(pred)
            pearson_coeff, p_value = pearsonr(true, pred)
            return pearson_coeff
        else:
            pearsons = []
            for dim in range(true.shape[-1]):
                pearson_coeff, p_value = pearsonr(true[:, dim], pred[:, dim])
                pearsons.append(pearson_coeff)
            return pearsons

    @staticmethod
    def spearman(true, pred):
        if true.shape[-1] == 1:
            true, pred = np.squeeze(true), np.squeeze(pred)
            spearman_coeff, p_value = spearmanr(true, pred)
            return spearman_coeff
        else:
            spearmans = []
            for dim in range(true.shape[-1]):
                spearman_coeff, p_value = spearmanr(true[:, dim], pred[:, dim])
                spearmans.append(spearman_coeff)
            return spearmans


    # @staticmethod
    # def correlation_matrix(true, pred):
    #     for ix in range(true.shape[0]):

    def __call__(self, true, pred, kinds):
        metrics = {}
        for kind in kinds:
            try:
                    fn = getattr(self, kind)
            except NameError as e:
                    print(e)
            error = fn(true, pred)
            metrics[kind] = error
        return metrics

    @staticmethod
    def get_best_metric(kind, metric_list):
        '''
        retrieve the dictionary for which the metric is the best
        '''
        if kind == 'r2':
            r2s = [d['r2'] for d in metric_list]
            max_ix = np.argmax(r2s)
            return metric_list[max_ix]
        elif kind == 'rmse':
            rmses = [d['rmse'] for d in metric_list]
            min_ix = np.argmin(rmses)
            return metric_list[min_ix]
        elif kind == 'mae':
            maes = [d['mae'] for d in metric_list]
            min_ix = np.argmin(maes)
            return metric_list[min_ix]
        elif kind == 'pearson':
            pearsons = [d['pearson'] for d in metric_list]
            max_ix = np.argmax(pearsons)
            return metric_list[max_ix]
        elif kind == 'spearman':
            spearmans = [d['spearman'] for d in metric_list]
            max_ix = np.argmax(spearmans)
            return metric_list[max_ix]
        else:
            raise NotImplementedError

    @staticmethod
    def get_all_best_metric(types, metrics_list):
        '''
        Retrieve all the best metrics
        '''
        best = {}
        for metric in types:
            met = [d[metric] for d in metrics_list]
            if metric in ['r2', 'spearman', 'pearson']:
                b = np.amax(met)
            elif metric in ['rmse', 'mae']:
                b = np.amin(met)
            best[metric] = b
        return best

    @staticmethod
    def get_best_index(kind, metric_list):
        '''
        retrieve training index for which the best metric
        is reported
        '''
        if kind == 'r2':
            r2s = [d['r2'] for d in metric_list]
            max_ix = np.argmax(r2s)
            return max_ix
        elif kind == 'rmse':
            rmses = [d['rmse'] for d in metric_list]
            min_ix = np.argmin(rmses)
            return min_ix
        elif kind == 'mae':
            maes = [d['mae'] for d in metric_list]
            min_ix = np.argmin(maes)
            return min_ix
        elif kind == 'pearson':
            pearsons = [d['pearson'] for d in metric_list]
            max_ix = np.argmax(pearsons)
            return max_ix
        elif kind == 'spearman':
            spearmans = [d['spearman'] for d in metric_list]
            max_ix = np.argmax(spearmans)
            return max_ix
        else:
            raise NotImplementedError


    @staticmethod
    def early_stopping(kind, metrics_list, patience):
        '''
        If patience is set to None, only stop training
        once we have seen the maximum number of epochs
        '''
        stop = False
        if patience == None:
            return stop
        else:
            if kind == 'r2':
                r2s = [d['r2'] for d in metrics_list]
                best_ix = np.argmax(r2s)
                if len(metrics_list) - best_ix > patience:
                    stop = True
            elif kind == 'rmse':
                r2s = [d['rmse'] for d in metrics_list]
                best_ix = np.argmin(r2s)
                if len(metrics_list) - best_ix > patience:
                    stop = True
            return stop
