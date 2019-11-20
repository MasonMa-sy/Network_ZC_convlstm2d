"""
for every @interval month, predict 1 to @prediction_month month, get nino34 index, and plot.
it's just like the way that LDEO model often uses it. http://ocp.ldgo.columbia.edu/~dchen/forecast.html
add seasonal circle 0304
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
from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation
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


model_name = name_list.model_name
model_type = name_list.model_type
is_seasonal_circle = name_list.is_seasonal_circle
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
elif model_type == 'dense':
    model = load_model('..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error
        , 'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error})

# Predict
# file_num = np.arange(3100, 3400, 30)
# for x in file_num:
#     predict_data = np.empty([1, 540])
#     predict_data[0] = data_loader.read_data(x)
#     data_loader.write_data(x, model.predict(predict_data)[0])
file_num = 100
month = 100
interval = 12
prediction_month = 12
data_preprocess_method = name_list.data_preprocess_method
# for dense_model
# predict_data = np.empty([1, 540])
# predict_data[0] = file_helper.read_data(file_num)
# file_helper.write_data(file_num, model.predict(predict_data)[0])

# for dense_model ssta and ha
predict_data = np.empty([1, 20, 27, 2])
data_y = np.empty([1, 20, 27, 2])
if is_seasonal_circle:
    data_sc = np.empty([1, 1], dtype='int32')
if model_type == 'conv':
    data_x = np.empty([1, 20, 27, 2])
elif model_type == 'dense':
    data_x = np.empty([1, 1080])

for start_month in range(file_num, file_num + month - interval, interval):
    predict_data[0] = file_helper_unformatted.read_data_sstaha(start_month)
    nino34 = [index_calculation.get_nino34(predict_data[0])]

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
    elif model_type == 'dense':
        data_x[0] = np.reshape(predict_data[0], (1, 1080))

    for i in range(prediction_month):
        if is_seasonal_circle:
            data_sc[0] = [(start_month+i) % 12]
            data_x = model.predict([data_x, data_sc])
        else:
            data_x = model.predict(data_x)
        # data_x = model.predict(data_x)

        if model_type == 'conv':
            data_y[0] = data_x[0]
        elif model_type == 'dense':
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

        # calculate nino 3.4 index
        nino34_temp1 = index_calculation.get_nino34(data_y[0])
        nino34.append(nino34_temp1)
    # file_helper_unformatted.write_data(file_num+month, data_temp[1])
    x = np.linspace(start_month, start_month + prediction_month, prediction_month + 1)
    plt.plot(x, nino34, 'b')
x = np.linspace(file_num, file_num + month, month + 1)
plt.plot(x, index_calculation.get_nino34_from_data(file_num, month), 'r', linewidth=1)
# plt.legend(['prediction', 'ZCdata'], loc='upper right')
plt.show()


# file_helper_unformatted.write_data(file_num, model.predict(data_x)[0])

# for convolutional model only ssta
# predict_data = np.empty([1, 540])
# predict_data[0] = file_helper.read_data(file_num)
# predict_result = model.predict(np.reshape(predict_data[0], (1, 20, 27, 1)))
# file_helper.write_data_conv2d(file_num, predict_result[0])

# for convolutional model ssta and ha
# predict_data = np.empty([1, 20, 27, 2])
# predict_data[0] = file_helper.read_data_sstaha(file_num)
# predict_result = model.predict(predict_data)
# file_helper.write_data(file_num, predict_result[0])
