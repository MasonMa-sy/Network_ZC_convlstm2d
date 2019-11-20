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
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list, math_tool
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
    return np.sum((y_true-y_pred)**2)/y_true.size


model_name = name_list.model_name
model_type = name_list.model_type
is_retrain = name_list.is_retrain
is_seasonal_circle = name_list.is_seasonal_circle
if name_list.is_best:
    model_name = file_helper_unformatted.find_model_best(model_name)
# Load the model
if model_type == 'conv':
    kernel_size = name_list.kernel_size
    max_value = name_list.max_value
    ssim = DSSIMObjectiveCustom(kernel_size=kernel_size, max_value=max_value)
    ssim_metrics = DSSIMObjectiveCustom(kernel_size=7, max_value=10)

    def ssim_l1(y_true, y_pred):
        a = 0.84
        return a * ssim(y_true, y_pred) + (1 - a) * mean_absolute_error(y_true, y_pred)

    model = load_model('D:\msy\projects\zc\\Network\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
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
# phase = 'historical'
phase = 'ZC'
if phase == 'historical':
    file_num = 417
else:
    file_num = 10869
prediction_month = 1
start_month_length = 12
directly_month = 12
data_preprocess_method = name_list.data_preprocess_method
# for dense_model
# predict_data = np.empty([1, 540])
# predict_data[0] = file_helper.read_data(file_num)
# file_helper.write_data(file_num, model.predict(predict_data)[0])


def calculate_12_month(start_month_internal):
    # for dense_model ssta and ha
    predict_data = np.empty([1, 20, 27, 2])
    data_y = np.empty([1, 20, 27, 2])
    data_realistic = np.empty([1, 20, 27, 2])
    if is_seasonal_circle:
        data_sc = np.empty([1, 1], dtype='int32')
    if model_type == 'conv':
        data_x = np.empty([1, 20, 27, 2])
    else:
        data_x = np.empty([1, 1080])
    predict_data[0] = file_helper_unformatted.read_data_sstaha(start_month_internal)
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

        if is_seasonal_circle:
            data_sc[0] = [(start_month+i) % 12]
            data_x = model.predict([data_x, data_sc])
        else:
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
        data_realistic[0] = file_helper_unformatted.read_data_sstaha(start_month+i+1)
        if is_retrain:
            data_realistic = file_helper_unformatted.exchange_rows(data_realistic)
        ssta_error[i] = calculate_error(data_y[0], data_realistic[0])
        # ssta_error[i] = math_tool.calculate_rmse(data_y[0], data_realistic[0])
    print(ssta_error)
    ssta_error2 = ssta_error.copy()
    for i in range(1, directly_month):
        ssta_error2[i] = (ssta_error[i] - ssta_error[i-1])
    ssta_error2[0] = ssta_error2[0]
    return ssta_error2


if __name__ == '__main__':
    ssta_error_all = np.empty([start_month_length, directly_month])
    for i1 in range(0, start_month_length):
        start_month = file_num + i1
        ssta_error_all[i1] = calculate_12_month(start_month)
    print(ssta_error_all)
    x = np.linspace(0, 11, 12, dtype=int)
    y = np.linspace(0, 11, 12, dtype=int)
    fig, ax = plt.subplots()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.xticks(x, x+1)
    plt.yticks(y, ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
    # v = np.linspace(-20, 70, 10)
    plt.contourf(x, y, ssta_error_all)
    # plt.contour(x, y, ssta_error_all, 20)
    plt.colorbar()
    if phase == 'historical':
        plt.plot(x, 9 - x, 'blue', linewidth=2)
    # plt.plot(x, 13-x, 'r', linewidth=2)
    # plt.plot(x, 14 - x, 'r', linewidth=2)
    # plt.plot(x, 15 - x, 'r', linewidth=2)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15}
    if phase == 'historical':
        ax.set_title('MSE growth rates of CNN-TRANS', fontsize=15, fontname='Times New Roman')
    else:
        ax.set_title('MSE growth rates of CNN-FREE', fontsize=15, fontname='Times New Roman')
    ax.set_ylabel('Start month', font1)
    ax.set_xlabel('Lead time(month)', font1)
    plt.set_cmap('Reds')
    plt.show()

# print(ssta_error)
# print(ssta_error[2]-0)
# print(ssta_error[5]-ssta_error[2])
# print(ssta_error[8]-ssta_error[5])
# print(ssta_error[11]-ssta_error[8])

