#!/usr/bin/env python

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

#==============================================================================

ACT_FUNCS = {
    'leaky_relu': lambda y: tf.nn.leaky_relu(y, 0.2),
    'linear': lambda y: y,
    'relu': lambda y: tf.nn.relu(y),
    'sigmoid': lambda y: tf.nn.sigmoid(y),
    'softmax': lambda y: tf.nn.softmax(y),
    'softplus': lambda y: tf.nn.softplus(y),
    'softsign': lambda y: tf.nn.softsign(y),
    'lambda': lambda y: y,}

LAYERS = {
    'tf_dense': tf.keras.layers.Dense,
    'tfp_dense': tfp.layers.DenseReparameterization,
    'tfp_dense_local': tfp.layers.DenseLocalReparameterization,
    'lambda': tf.keras.layers.Lambda,}


def parse_feature_vars(vars):
    bn1    = eval_batch_norm(vars[:4])
    dense1 = eval_dense(vars[4:6])
    bn2    = eval_batch_norm(vars[6:10])
    dense2 = eval_dense(vars[10:12])
    feature_vars =  {'bn1': bn1, 'dense1': dense1, 'bn2': bn2, 'dense2': dense2}
    return feature_vars


def parse_latent_vars(vars):
    bn1    = eval_batch_norm(vars[:4])
    dense1 = eval_dense(vars[4:6])
    bn2    = eval_batch_norm(vars[6:10])
    dense2 = eval_dense(vars[10:12])
    bn3    = eval_batch_norm(vars[12:16])
    dense3 = eval_dense(vars[16:18])
    bn4    = eval_batch_norm(vars[18:22])
    dense4 = eval_dense(vars[22:24])
    latent_vars = {'bn1': bn1, 'dense1': dense1, 'bn2': bn2, 'dense2': dense2,
                   'bn3': bn3, 'dense3': dense3, 'bn4': bn4, 'dense4': dense4}
    return latent_vars

def parse_target_vars(vars):
    bn1    = eval_batch_norm(vars[:4])
    dense1 = eval_dense(vars[4:6])
    bn2    = eval_batch_norm(vars[6:10])
    dense2 = eval_dense(vars[10:12])
    target_vars =  {'bn1': bn1, 'dense1': dense1, 'bn2': bn2, 'dense2': dense2}
    return target_vars

def eval_batch_norm(vars):
    gamma = vars[0].eval()
    beta  = vars[1].eval()
    moving_mean = vars[2].eval()
    moving_var  = vars[3].eval()
    return gamma, beta, moving_mean, moving_var

def eval_dense(vars):
    kernel = vars[0].eval()
    bias   = vars[1].eval()
    return kernel, bias
