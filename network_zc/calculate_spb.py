"""
this script not only plot nino34 index, but also generate output file.
"""
# Third-party libraries
from keras.layers import Input, Dense
from keras import optimizers
from keras.models import Model
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
# My libraries
from network_zc.tools import file_helper_unformatted,data_preprocess, name_list
from network_zc.tools import index_calculation
from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.model_trainer import dense_trainer
from network_zc.model_trainer import dense_trainer_sstaha
from network_zc.model_trainer import dense_trainer_sstaha_4
from network_zc.model_trainer import convolutional_trainer_alex


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def calculate_error(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)


model_name = name_list.model_name
model_type = name_list.model_type
is_retrain = name_list.is_retrain
# Load the model
if model_type == 'conv':
    kernel_size = name_list.kernel_size
    max_value = name_list.max_value
    ssim = DSSIMObjectiveCustom(kernel_size=kernel_size, max_value=max_value)
    ssim_metrics = DSSIMObjectiveCustom(kernel_size=7, max_value=10)

    def ssim_l1(y_true, y_pred):
        a = 0.84
        return a * ssim(y_true, y_pred) + (1 - a) * mean_absolute_error(y_true, y_pred)

    model = load_model('..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
            'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error,
            'ssim_metrics': ssim_metrics, 'ssim_l1': ssim_l1, 'DSSIMObjective': ssim})
else:
    model = load_model('..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
            'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error})

# Predict
# file_num = np.arange(3100, 3400, 30)
# for x in file_num:
#     predict_data = np.empty([1, 540])
#     predict_data[0] = data_loader.read_data(x)
#     data_loader.write_data(x, model.predict(predict_data)[0])
file_num = 207
prediction_month = 1
directly_month = 12
data_preprocess_method = name_list.data_preprocess_method
# for dense_model
# predict_data = np.empty([1, 540])
# predict_data[0] = file_helper.read_data(file_num)
# file_helper.write_data(file_num, model.predict(predict_data)[0])

# for dense_model ssta and ha
predict_data = np.empty([1, 20, 27, 2])
data_y = np.empty([1, 20, 27, 2])
data_realistic = np.empty([1, 20, 27, 2])
if model_type == 'conv':
    data_x = np.empty([1, 20, 27, 2])
else:
    data_x = np.empty([1, 1080])
predict_data[0] = file_helper_unformatted.read_data_sstaha(file_num)
if is_retrain:
    predict_data = file_helper_unformatted.exchange_rows(predict_data)

# data preprocess z-zero
if data_preprocess_method == 'preprocess_Z':
    predict_data = data_preprocess.preprocess_Z(predict_data, 0)
# data preprocess dimensionless
if data_preprocess_method == 'dimensionless':
    redict_data = data_preprocess.dimensionless(predict_data, 0)
# data preprocess 0-1
if data_preprocess_method == 'preprocess_01':
    predict_data = data_preprocess.preprocess_01(predict_data, 0)
# data preprocess no month mean
if data_preprocess_method == 'nomonthmean':
    predict_data = data_preprocess.no_month_mean(predict_data, 0)

if model_type == 'conv':
    data_x[0] = predict_data[0]
else:
    data_x[0] = np.reshape(predict_data[0], (1, 1080))

ssta_error = np.empty([directly_month])
for i in range(directly_month):
    data_x = model.predict(data_x)

    if model_type == 'conv':
        data_y[0] = data_x[0]
    else:
        data_y[0] = np.reshape(data_x[0], (20, 27, 2))

    # data preprocess z-zero
    if data_preprocess_method == 'preprocess_Z':
        data_y = data_preprocess.preprocess_Z(data_y, 1)
    # data preprocess dimensionless
    if data_preprocess_method == 'dimensionless':
        data_y = data_preprocess.dimensionless(data_y, 1)
    # data preprocess 0-1
    if data_preprocess_method == 'preprocess_01':
        data_y = data_preprocess.preprocess_01(data_y, 1)
    # data preprocess no month mean
    if data_preprocess_method == 'nomonthmean':
        data_y = data_preprocess.no_month_mean(data_y, 1)
    data_realistic[0] = file_helper_unformatted.read_data_sstaha(file_num+i+1)
    if is_retrain:
        data_realistic = file_helper_unformatted.exchange_rows(data_realistic)
    ssta_error[i] = calculate_error(data_y[0], data_realistic[0])

print(ssta_error)
print(ssta_error[2]-0)
print(ssta_error[5]-ssta_error[2])
print(ssta_error[8]-ssta_error[5])
print(ssta_error[11]-ssta_error[8])

