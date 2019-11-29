from keras.layers import Lambda
from keras import backend as K
import numpy as np


def function(tensor):
    # print(tensor.shape)
    nino = np.zeros(tensor.shape[1:])
    for i in range(7, 13):
        for j in range(11, 20):
            nino[i][j][0] = 1
    nino = K.constant(nino)
    nino = K.expand_dims(nino, axis=0)
    nino = tensor * nino
    return K.sum(nino, axis=[1, 2])/54


def function2(tensor):
    # print(tensor.shape)
    nino = np.zeros(tensor.shape[1:])
    for i in range(7, 13):
        for j in range(11, 20):
            nino[i][j][0] = 1
    nino = K.constant(nino)
    nino = K.expand_dims(nino, axis=0)
    nino = tensor * nino
    return K.sum(nino, axis=[1, 2])/54


def output_shape(input_shape):
    return (input_shape[0],) + (1,)


def get_nino_layer():
    layer = Lambda(function, output_shape)
    return layer
