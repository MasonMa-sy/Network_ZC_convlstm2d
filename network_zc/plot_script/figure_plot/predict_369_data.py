"""
For final paper plot and analysis
According to model_name and directly_month.
Generate prediction data
"""
# Third-party libraries
from keras.layers import Input, Dense
from keras import optimizers
from keras.models import Model
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import h5py
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


# file_num = 10860
# month = 100
file_num = 417
month = 47
interval = 1
prediction_month = 1
directly_month = 12
data_preprocess_method = name_list.data_preprocess_method
predict_file_dir = name_list.predict_file_dir

if __name__ == '__main__':
    model_name = name_list.model_name
    if name_list.is_best:
        model_name = file_helper_unformatted.find_model_best(model_name)
    model_type = name_list.model_type
    is_retrain = name_list.is_retrain
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

        model = load_model('..\..\..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
                'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error,
                'ssim_metrics': ssim_metrics, 'ssim_l1': ssim_l1, 'DSSIMObjective': ssim})
    elif model_type == 'dense':
        model = load_model('..\..\..\model\\' + model_name + '.h5', custom_objects={'mean_squared_error': mean_squared_error,
                'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error})

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

    # make directory
    file_path = predict_file_dir + model_name + '\\' + str(directly_month) + '\\'
    file_helper_unformatted.mkdir(file_path)

    predict_data = np.empty([1, 20, 27, 2])
    data_y = np.empty([1, 20, 27, 2])
    if is_seasonal_circle:
        data_sc = np.empty([1, 1], dtype='int32')
    if model_type == 'conv':
        data_x = np.empty([1, 20, 27, 2])
    else:
        data_x = np.empty([1, 1080])
    nino34 = []
    for start_month in range(file_num, file_num+month+1, interval):
        predict_data[0] = file_helper_unformatted.read_data_sstaha(start_month)
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
        elif model_type == 'dense':
            data_x[0] = np.reshape(predict_data[0], (1, 1080))

        for i in range(directly_month):
            if is_seasonal_circle:
                data_sc[0] = [(start_month+i) % 12]
                data_x = model.predict([data_x, data_sc])
            else:
                data_x = model.predict(data_x)

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

        if is_retrain:
            data_y = file_helper_unformatted.exchange_rows(data_y)

        file_helper_unformatted.write_data_best(file_path, start_month+directly_month*prediction_month, data_y)
        # calculate nino 3.4 index
        nino34_temp1 = index_calculation.get_nino34(data_y[0])
        nino34.append(nino34_temp1)
        # file_helper_unformatted.write_data(file_num+month, data_temp[1])
    # x = np.linspace(file_num, start_month + prediction_month, prediction_month + 1)
    x = np.linspace(file_num+prediction_month*directly_month, file_num+month+directly_month*prediction_month, month+1)
    plt.plot(x, nino34, 'b')
    nino34_from_data = index_calculation.get_nino34_from_data(file_num, month)
    plt.plot(x, nino34_from_data, 'r', linewidth=1)
    print(math_tool.pearson_distance(nino34, nino34_from_data))
    # plt.legend(['prediction', 'ZCdata'], loc='upper right')
    f = h5py.File(file_path+'nino34.h5', 'w')
    f.create_dataset('X', data=x)
    f.create_dataset('Y', data=nino34)
    f.close()
    plt.show()
