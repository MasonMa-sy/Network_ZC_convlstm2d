"""
This script is done for load SST data from .dat.
The data is binary.
"""

# Third-party libraries
import numpy as np
import struct
# My libraries
from network_zc.tools import name_list

data_file = name_list.data_file
data_name = name_list.data_name
data_file_statistics = name_list.data_file_statistics
data_preprocessing_file = name_list.data_preprocessing_file


def load_sst_for_dense(training_num, testing_num=0):
    """
    Return one line data of sst for dense model.
    The data form 0 to train_num is training data.If validation_num
    TODO for Notes
    :param training_num:
    :param testing_num:
    :return:
    """
    training_data = np.empty([training_num + 1, 540])
    num = 0
    while num <= training_num:
        training_data[num] = read_data(num)
        num += 1
    testing_data = np.empty(1)
    if testing_num != 0:
        testing_data = np.empty([testing_num - training_num, 540])
        num = training_num + 1
        while num <= testing_num:
            testing_data[num] = read_data(num)
            num += 1
    return training_data, testing_data


def load_sst_for_conv2d(training_num, testing_num=0):
    """
    Return [training_num+1, 20, 27, 1] data of sst for conv2d model.
    The data form 0 to train_num is training data.If validation_num
    TODO for Notes
    :param training_num:
    :param testing_num:
    :return:
    """
    training_data, testing_data = load_sst_for_dense(training_num, testing_num)
    training_data = np.reshape(training_data, (training_num + 1, 20, 27, 1))
    if testing_num != 0:
        testing_data = np.reshape(testing_data, (training_num + 1, 20, 27, 1))
    return training_data, testing_data


def read_data(num):
    """
    Read ssta data of NO.num
    :param num:
    :return: One-dimensional array,length 540
    """
    data_file_change = name_list.data_file
    data_name_change = name_list.data_name
    file_num = "%05d" % num
    filename = "D:\\files\课题组\zc_network\zc_data\\" + data_file_change + data_name_change + file_num + ".dat"
    fh = open(filename, mode='r')
    list_temp = []
    for line in fh:
        list_temp.append(line)
    fh.close()
    count = 0
    sst = []
    while count < 25:
        if count < 5:
            count = count + 1
            continue
        list_temp2 = list(map(float, list_temp[count].split()))
        count = count + 1
        sst.extend(list_temp2[5:32])
    return np.array(sst)


def load_sstha_for_conv2d(training_start, training_num, testing_num=0):
    """
    Return [training_num+1, 20, 27, 2] data of ssta and ha for conv2d model.
    The data form training_start to train_num is training data.If validation_num
    TODO for Notes
    :param training_start:
    :param training_num:
    :param testing_num:now this is not used
    :return:
    """
    training_data = np.empty([training_num + 1 - training_start, 20, 27, 2])
    num = training_start
    while num <= training_num:
        training_data[num - training_start] = read_data_sstaha(num)
        num += 1
    testing_data = np.empty(1)
    if testing_num != 0:
        testing_data = np.empty([testing_num - training_num, 20, 27, 2])
        num = training_num + 1
        while num <= testing_num:
            testing_data[num] = read_data_sstaha(num)
            num += 1
    return training_data, testing_data


def load_sstha_for_conv2d_separate(training_num):
    """
    Return [training_num, 20, 27, 2],[training_num, 20, 27, 2] traing and testing
        data of ssta and ha for conv2d model.
    The data form 0 to train_num is training data.If validation_num
    TODO for Notes
    :param training_num:
    :return:
    """
    training_data = np.empty([training_num, 20, 27, 2])
    testing_data = np.empty([training_num, 20, 27, 2])
    num = 0
    while num < training_num:
        training_data[num] = read_data_sstaha_separate(num, 1)
        testing_data[num] = read_data_sstaha_separate(num, 2)
        num += 1
    return training_data, testing_data


def read_data_sstaha_separate(num, type_num):
    """
    Read ssta and ha data of NO.num
    :param type_num: 1 for training, 2 for testing.
    :param num: the NO. of file name
    :return: One-dimensional array,length 540
    """
    file_num = "%05d" % num
    filename = ''
    if type_num == 1:
        filename = "D:\\files\课题组\zc_network\zc_data\\" + data_file + "\data_in_" + file_num + ".dat"
    if type_num == 2:
        filename = "D:\\files\课题组\zc_network\zc_data\\" + data_file + "\data_out_" + file_num + ".dat"
    fh = open(filename, mode='r')
    list_temp = []
    for line in fh:
        list_temp.append(line)
    fh.close()
    count = 0
    sst = []
    while count < 55:
        if count < 5 or (24 < count < 35):
            count = count + 1
            continue
        list_temp2 = list(map(float, list_temp[count].split()))
        count = count + 1
        sst.extend(list_temp2[5:32])
    data = np.array(sst)
    ssta = data[:540]
    ha = data[540:]
    training_data = np.empty([20, 27, 2])
    training_data[:, :, 0] = np.reshape(ssta, (20, 27))
    training_data[:, :, 1] = np.reshape(ha, (20, 27))
    return training_data


def read_data_sstaha(num):
    """
    Read ssta and ha data of NO.num
    :param num:
    :return: One-dimensional array,length 540
    """
    data_file_change = name_list.data_file
    data_name_change = name_list.data_name
    file_num = "%05d" % num
    filename = "D:\\files\课题组\zc_network\zc_data\\" + data_file_change + data_name_change + file_num + ".dat"
    fh = open(filename, mode='rb')
    training_data = np.empty([20, 27, 2])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][0] = text
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][1] = text
    fh.close()
    return training_data


def read_data_best(file_path, num):
    """
    For final paper plot and analysis
    Read ssta and ha data of NO.num
    :param file_path:
    :param num:
    :return: two-dimensional array,length 540
    """
    file_num = "%05d" % num
    filename = file_path + '\data_' + file_num + ".dat"
    fh = open(filename, mode='rb')
    training_data = np.empty([20, 27, 2])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][0] = text
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][1] = text
    fh.close()
    return training_data


def read_data_historical(file_path, num):
    """
    For final paper plot and analysis
    Read ssta and ha data of NO.num
    :param file_path:
    :param num:
    :return: two-dimensional array,length 540
    """
    file_num = "%05d" % num
    filename = file_path + '\data_historical_' + file_num + ".dat"
    fh = open(filename, mode='rb')
    training_data = np.empty([20, 27, 2])
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][0] = text
    for i in range(20):
        for j in range(27):
            data = fh.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            training_data[i][j][1] = text
    fh.close()
    return training_data


def write_data(num, data):
    """
    Write data named num containing data for dense.
    :param num:
    :param data:
    :return:
    """
    file_num = "%05d" % num
    filename = data_file_statistics + file_num + ".dat"
    data = np.reshape(data, (20, 27, 2))
    with open(filename, 'wb') as fp:
        #    data = fp.read(4)
        for i in range(20):
            for j in range(27):
                temp = struct.pack("d", data[i][j][0])
                fp.write(temp)
        for i in range(20):
            for j in range(27):
                temp = struct.pack("d", data[i][j][1])
                fp.write(temp)
    fp.close()


def write_data_best(file_path, num, data):
    """
    For final paper plot and analysis
    Write data named num containing data for dense. If the directory is not exist, it will be made.
    :param file_path:
    :param num:
    :param data:
    :return:
    """
    file_num = "%05d" % num
    filename = file_path + '\data_' + file_num + ".dat"
    data = np.reshape(data, (20, 27, 2))
    with open(filename, 'wb') as fp:
        #    data = fp.read(4)
        for i in range(20):
            for j in range(27):
                temp = struct.pack("d", data[i][j][0])
                fp.write(temp)
        for i in range(20):
            for j in range(27):
                temp = struct.pack("d", data[i][j][1])
                fp.write(temp)
    fp.close()


def write_data_conv2d(num, data):
    """
    Write data named num containing data for conv2d.
    :param num:
    :param data:
    :return:
    """
    file_num = "%05d" % num
    filename = "..\data\predict_data_" + file_num + ".dat"
    fh = open(filename, mode='w')
    i = 0
    while i < 5:
        j = 0
        while j < 34:
            fh.write("%13.5f" % 0.0)
            j += 1
        i += 1
        fh.write('\n')
    i = 0
    data_row = 0
    while i < 20:
        j = 0
        while j < 5:
            fh.write("%13.5f" % 0.0)
            j += 1
        data_i = 0
        while data_i < 27:
            fh.write("%13.5f" % float(data.item((data_row, data_i, 0))))
            data_i += 1
        data_row += 1
        j = 0
        while j < 2:
            fh.write("%13.5f" % 0.0)
            j += 1
        fh.write('\n')
        i += 1
    i = 0
    while i < 5:
        j = 0
        while j < 34:
            fh.write("%13.5f" % 0.0)
            j += 1
        i += 1
        fh.write('\n')
    fh.close()


def for_decay_trainingdata(training_data):
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + "\decay_num.dat"
    fh = open(filename, mode='r')
    for line in fh:
        training_data[int(line)] *= 10
    fh.close()
    return training_data


def find_logs(filename):
    file = '..\..\model\logs\\' + filename
    return file


def find_logs_final(filename):
    file = '..\..\model\\best\\logs\\' + filename
    return file


def find_model_best(modelname):
    file = 'best\\' + modelname
    return file


def exchange_rows(training_data):
    exchange_data = training_data.copy()
    rows = training_data.shape[1]
    is_sequence = name_list.is_sequence
    for i in range(rows):
        if not is_sequence:
            exchange_data[:, i, :, :] = training_data[:, rows-i-1, :, :]
        else:
            exchange_data[:, :, i, :, :] = training_data[:, :, rows - i - 1, :, :]
    return exchange_data


def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    is_exists = os.path.exists(path)
    # 判断结果
    if not is_exists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' was created')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' is existed')
        return False
