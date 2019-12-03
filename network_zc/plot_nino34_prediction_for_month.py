"""
for every @interval month, predict @prediction_month * @directly_month nino3.4 index,
plot and calculate correlation coefficient.
add seasonal circle 0225
"""
# Third-party libraries
from keras.layers import Input, Dense, LeakyReLU
from keras import optimizers
from keras.models import Model
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
# My libraries
from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation, math_tool
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


is_retrain = name_list.is_retrain
if is_retrain:
    model_name = name_list.retrain_model_name
else:
    model_name = name_list.model_name
if name_list.is_best:
    model_name = file_helper_unformatted.find_model_best(model_name)
model_type = name_list.model_type

is_seasonal_circle = name_list.is_seasonal_circle
lrelu = lambda y: LeakyReLU(alpha=0.3)(y)
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
                                                                          'root_mean_squared_error': root_mean_squared_error,
                                                                          'mean_absolute_error': mean_absolute_error,
                                                                          '<lambda>': lrelu})
elif model_type == 'dense':
    model = load_model('..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
                                                                          'root_mean_squared_error': root_mean_squared_error,
                                                                          'mean_absolute_error': mean_absolute_error})

# Predict
# file_num = np.arange(3100, 3400, 30)
# for x in file_num:
#     predict_data = np.empty([1, 540])
#     predict_data[0] = data_loader.read_data(x)
#     data_loader.write_data(x, model.predict(predict_data)[0])
"""
file_num: the first prediction of data num
month: month num to predict
interval: Prediction interval
prediction_month: For the model to predict month num
directly_month: rolling run model times
"""
file_num = 417
month = 47
# file_num = 340
# month = 124
interval = 1
prediction_month = name_list.prediction_month
directly_months = [3, 6, 9, 12]
# directly_months = [1, 2, 3, 4]
data_preprocess_method = name_list.data_preprocess_method
is_sequence = name_list.is_sequence
time_step = name_list.time_step
is_nino_output = name_list.is_nino_output
# for dense_model
# predict_data = np.empty([1, 540])
# predict_data[0] = file_helper.read_data(file_num)
# file_helper.write_data(file_num, model.predict(predict_data)[0])

# for dense_model ssta and ha
nino34_from_data = index_calculation.get_nino34_from_data(file_num + time_step, month - time_step)
# print(nino34_from_data)
for di, directly_month in enumerate(directly_months):
    if not is_sequence:
        predict_data = np.empty([1, 20, 27, 2])
        if model_type == 'conv':
            data_x = np.empty([1, 20, 27, 2])
        else:
            data_x = np.empty([1, 1080])
    else:
        predict_data = np.empty([1, time_step, 20, 27, 2])
        if is_seasonal_circle:
            pass
        else:
            pass
        data_x = np.empty([1, time_step, 20, 27, 2])
        data_yy = np.empty([1, 20, 27, 2])
    if is_seasonal_circle:
        data_sc = np.empty([1, 1], dtype='int32')
    data_y = np.empty([1, 20, 27, 2])
    if is_nino_output:
        nino34_output = []
        rmse_sst = []
        nino_temp = 0
    nino34 = []
    # The time step for non-sequence prediction should be 1, but it was originally written as 0.
    # So it caused the inconsistency.
    for start_month in range(file_num - prediction_month * directly_month,
                             file_num + month - prediction_month * directly_month + 1 - time_step, interval):
        if not is_sequence:
            predict_data[0] = file_helper_unformatted.read_data_sstaha(start_month)
            predict_data = data_preprocess.data_preprocess(predict_data, 0)
        else:
            for i in range(time_step):
                predict_data[0][i] = file_helper_unformatted.read_data_sstaha(start_month + i + 1)
            predict_data[0] = data_preprocess.data_preprocess(predict_data[0], 0)
        if is_retrain:
            predict_data = file_helper_unformatted.exchange_rows(predict_data)

        if model_type == 'conv':
            data_x[0] = predict_data[0]
        elif model_type == 'dense':
            data_x[0] = np.reshape(predict_data[0], (1, 1080))

        for i in range(directly_month):
            if not is_sequence:
                if is_seasonal_circle:
                    data_sc[0] = [(start_month + i) % 12]
                    data_x = model.predict([data_x, data_sc])
                else:
                    data_x = model.predict(data_x)
            else:
                if is_seasonal_circle:
                    if not is_nino_output:
                        data_sc[0] = [(start_month + i + 1) % 12]
                        data_yy = model.predict([data_x, data_sc])
                        data_x[0] = np.concatenate((data_x[0], data_yy), axis=0)[1:]
                    else:
                        data_sc[0] = [(start_month + i + 1) % 12]
                        data_yy, nino_temp = model.predict([data_x, data_sc])
                        data_x[0] = np.concatenate((data_x[0], data_yy), axis=0)[1:]
                else:
                    data_yy = model.predict(data_x)
                    data_x[0] = np.concatenate((data_x[0], data_yy), axis=0)[1:]
        if model_type == 'conv':
            if not is_sequence:
                data_y[0] = data_x[0]
            else:
                data_y[0] = data_yy[0]
        elif model_type == 'dense':
            data_y[0] = np.reshape(data_x[0], (20, 27, 2))

        data_y = data_preprocess.data_preprocess(data_y, 1)

        # calculate nino 3.4 index
        nino34_temp1 = index_calculation.get_nino34(data_y[0])
        nino34.append(nino34_temp1)
        if is_nino_output:
            nino34_output.append(nino_temp[0][0]*2)
            data_from_data = file_helper_unformatted.read_data_sstaha(start_month+time_step)
            rmse_sst.append(math_tool.calculate_rmse(data_y[0, :, :, 0], data_from_data[:, :, 0]))
        # file_helper_unformatted.write_data(file_num+month, data_temp[1])
    # x = np.linspace(file_num, start_month + prediction_month, prediction_month + 1)
    x = np.linspace(file_num + time_step, file_num + month, month + 1 - time_step)
    plt.subplot(221 + di)
    plt.plot(x, nino34, 'b')
    plt.plot(x, nino34_from_data, 'r', linewidth=1)
    # if is_nino_output:
    #     plt.plot(x, nino34_output, 'g', linewidth=1)
    print(math_tool.pearson_distance(nino34, nino34_from_data), end=' ')
    print(math_tool.calculate_rmse(nino34, nino34_from_data), end=' ')
    print(sum(rmse_sst)/len(rmse_sst))

    # constant = [0 for _ in range(len(nino34_from_data))]
    # print('@', math_tool.pearson_distance(constant, nino34_from_data))
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
