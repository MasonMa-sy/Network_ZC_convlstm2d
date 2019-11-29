"""
This script is done for data preprocess
"""

# Third-party libraries
import numpy as np
import struct
# My libraries
from network_zc.tools import file_helper_unformatted, name_list

data_file = name_list.data_file
data_name = name_list.data_name
data_file_statistics = name_list.data_file_statistics
data_preprocessing_file = name_list.data_preprocessing_file


def read_preprocessing_data():
    """

    :return: mean data and std data for data preprocessing.
    """
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "mean.dat"
    fh = open(filename, mode='rb')
    data_mean = np.empty([20, 27, 2])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            data_mean[i][j][0] = text
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            data_mean[i][j][1] = text
    fh.close()
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "std.dat"
    fh = open(filename, mode='rb')
    data_std = np.empty([20, 27, 2])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            data_std[i][j][0] = text
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            data_std[i][j][1] = text
    fh.close()
    return data_mean, data_std


def preprocess_Z(training_data, di_or_de):
    """
    for Z-score normalization
    :param di_or_de:
    :param training_data:
    :return:
    """
    data_mean, data_std = read_preprocessing_data()
    if di_or_de == 0:
        training_data = (training_data - data_mean) / data_std
    if di_or_de == 1:
        training_data = training_data*data_std + data_mean
    return training_data


def preprocess_01_for_train(training_data):
    training_min0 = training_data[:, :, :, 0].min()
    training_max0 = training_data[:, :, :, 0].max()
    training_min1 = training_data[:, :, :, 1].min()
    training_max1 = training_data[:, :, :, 1].max()
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "MaxMin.dat"
    fh = open(filename, mode='wb')
    data = struct.pack("d", training_min0)
    fh.write(data)
    data = struct.pack("d", training_max0)
    fh.write(data)
    data = struct.pack("d", training_min1)
    fh.write(data)
    data = struct.pack("d", training_max1)
    fh.write(data)
    fh.close()


def preprocess_01(training_data, di_or_de):
    """
    for 0-1 normalization
    :param di_or_de:
    :param training_data:
    :return:
    """
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "MaxMin.dat"
    fh = open(filename, mode='rb')
    data = fh.read(8)
    training_min0 = struct.unpack("d", data)[0]
    data = fh.read(8)
    training_max0 = struct.unpack("d", data)[0]
    data = fh.read(8)
    training_min1 = struct.unpack("d", data)[0]
    data = fh.read(8)
    training_max1 = struct.unpack("d", data)[0]
    fh.close()
    # print("%g!%g!%g!%g" % (training_min0, training_max0, training_min1, training_max1))
    if di_or_de == 0:
        training_data[:, :, :, 0] = (training_data[:, :, :, 0] - training_min0) / (training_max0 - training_min0)
        training_data[:, :, :, 1] = (training_data[:, :, :, 1] - training_min1) / (training_max1 - training_min1)
    if di_or_de == 1:
        training_data[:, :, :, 0] = training_data[:, :, :, 0] * (training_max0 - training_min0) + training_min0
        training_data[:, :, :, 1] = training_data[:, :, :, 1] * (training_max1 - training_min1) + training_min1
    return training_data


def dimensionless(training_data, di_or_de):
    """
    ssta/2, h1a/50
    :param training_data:
    :param di_or_de:0 for many sample /, 1 for many sample *.
    :return:
    """
    if di_or_de == 0:
        training_data[:, :, :, 0] = training_data[:, :, :, 0] / 2
        training_data[:, :, :, 1] = training_data[:, :, :, 1] / 50
    if di_or_de == 1:
        training_data[:, :, :, 0] = training_data[:, :, :, 0] * 2
        training_data[:, :, :, 1] = training_data[:, :, :, 1] * 50
    return training_data


def sequence_data(data, input_length=3, prediction_month=1, output_sequence=False, validation_split=10):
    data_x = np.reshape([data[i: i + input_length] for i in range(data.shape[0]-input_length-(prediction_month-1))],
                        (-1, input_length)+data.shape[1:])
    data_y = np.reshape(data[input_length+prediction_month-1:], (-1,)+data.shape[1:])
    return data_x, data_y


def split_data(data, validation_split=10):
    pass


def data_preprocess(training_data, di_or_de, data_preprocess_method='dimensionless'):
    # data preprocess z-zero
    if data_preprocess_method == 'preprocess_Z':
        training_data = preprocess_Z(training_data, di_or_de)
    # data preprocess dimensionless
    if data_preprocess_method == 'dimensionless':
        training_data = dimensionless(training_data, di_or_de)
    # data preprocess 0-1
    if data_preprocess_method == 'preprocess_01':
        training_data = preprocess_01(training_data, di_or_de)
    # data preprocess no month mean
    if data_preprocess_method == 'nomonthmean':
        training_data = no_month_mean(training_data, di_or_de)
    return training_data


def read_sstm_data():
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "sstm.dat"
    fh = open(filename, mode='rb')
    data_sstm = np.empty([30, 34, 12])
    for l in range(12):
        for i in range(30):
            for j in range(34):
                data = fh.read(8)  # type(data) === bytes
                text = struct.unpack("d", data)[0]
                data_sstm[i][j][l] = text
    fh.close()
    data_sstm_region = np.empty([20, 27, 12])
    for l in range(12):
        for i in range(20):
            for j in range(27):
                data_sstm_region[i][j][l] = data_sstm[i+5][j+5][l]
    return data_sstm_region


def read_sst_mean_data():
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "sst_mean.dat"
    fh = open(filename, mode='rb')
    data_sst_mean = np.empty([20, 27])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            data_sst_mean[i][j] = text
    fh.close()
    return data_sst_mean


def no_month_mean(training_data, di_or_de):
    """
    First, add sstm. Then, subtract mean
    :param di_or_de: 2 for write sst_mean_data
    :param training_data:
    :return:
    """
    data_sstm = read_sstm_data()
    if di_or_de == 2:
        training_no_mean = training_data.copy()
        for i in range(training_data.shape[0]):
            training_no_mean[i, :, :, 0] = training_data[i, :, :, 0] + data_sstm[:, :, i % 12]
        data_mean = np.mean(training_no_mean[:, :, :, :], axis=0)
        file_helper_unformatted.write_data(1, data_mean)
        return
    data_mean = read_sst_mean_data()
    training_no_mean = training_data.copy()
    if di_or_de == 0:
        for i in range(training_data.shape[0]):
            training_no_mean[i, :, :, 0] = training_data[i, :, :, 0] + data_sstm[:, :, i % 12]
        training_no_mean[:, :, :, 0] = training_no_mean[:, :, :, 0] - data_mean
    if di_or_de == 1:
        training_no_mean[:, :, :, 0] = training_data[:, :, :, 0] + data_mean
        for i in range(training_data.shape[0]):
            training_no_mean[i, :, :, 0] = training_no_mean[i, :, :, 0] - data_sstm[:, :, i % 12]
    return training_no_mean
