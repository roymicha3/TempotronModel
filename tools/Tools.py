import numpy as np
import random
import os
from scipy.signal import convolve2d

def heaviside(time_step, time_cutoff, n):
    cutoff = int(time_cutoff // time_step)

    res = np.ones(int(n / time_step))
    res[:cutoff] = 0

    return res

def kFunction(T, time_step, ti):

    t = np.arange(0, T, time_step)

    res = heaviside(T, time_step, ti)
    res  = res * ( np.exp(-(t-ti)/ 3) - np.exp(-(t-ti) / 2) )
    
    return res

def numeric_integral(func, dt):
    return np.sum(func) * dt

def get_conv_idx(matrix_shape: tuple, window: np.array):
    indexes_matrix = np.arange(0, matrix_shape[0] * matrix_shape[1]).reshape(matrix_shape)
    conv_indexes = convolve2d(indexes_matrix, window, mode='valid').flatten().astype(int)
    pos_indexes = np.array([[i, j] for i in range(matrix_shape[0]) for j in range(matrix_shape[1])])

    return tuple(pos_indexes[conv_indexes].T.tolist())

def get_non_zero_idx(data: np.array):
    return np.nonzero(data).T