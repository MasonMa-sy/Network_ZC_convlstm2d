# Third-party libraries
import numpy as np
import struct
import matplotlib.pyplot as plt
# My libraries
from network_zc.tools import file_helper_unformatted


def get_nino34(data_2d):
    nino3_temp = 0
    for i in range(7, 13):
        for j in range(11, 20):
            nino3_temp += data_2d[i][j][0]
    return nino3_temp / 54


def get_nino34_from_data(file_num, month):
    nino3 = []
    for i in range(file_num, file_num + month + 1):
        data = file_helper_unformatted.read_data_sstaha(i)
        nino3.append(get_nino34(data))
    return nino3


def get_nino34_from_data_y(data_y):
    nino = np.zeros(data_y.shape[0:3])
    # print(nino.shape)
    for i in range(7, 13):
        for j in range(11, 20):
            nino[0][i][j] = 1
    for i in range(1, data_y.shape[0]):
        nino[i] = nino[0]
    # print(data_y[:, :, :, 0] * nino)
    return np.sum(data_y[:, :, :, 0] * nino, axis=(1, 2))/54


def plot_nino34():
    nino3 = []
    for num in range(1200):
        file_num = "%05d" % num
        meteo_file = "data_" + file_num + ".dat"

        count = 0
        sst = np.empty([20, 27])
        with open(meteo_file, 'rb') as fp:
            #    data = fp.read(4)
            for i in range(20):
                for j in range(27):
                    data = fp.read(8)  # type(data) === bytes
                    text = struct.unpack("d", data)[0]
                    sst[i][j] = text
        fp.close()
        nino3_temp = 0
        sumi = 0
        for i in range(7, 13):
            for j in range(11, 20):
                nino3_temp += sst[i][j]
                sumi = sumi + 1
        nino3.append(nino3_temp / sumi)
        # print(sumi)
    plt.plot(nino3)
    plt.show()
