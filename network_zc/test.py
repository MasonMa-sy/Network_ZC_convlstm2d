import numpy as np
# a = [[[i+(k*4+j)*3 for i in range(3)] for j in range(4)] for k in range(5)]
# print(a)
# print(np.sum(a, axis=1))
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=(0, 1)))
from network_zc.tools import data_preprocess, file_helper_unformatted, index_calculation

np.set_printoptions(threshold=np.inf)
training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(0, 464)
data_x, data_y = data_preprocess.sequence_data(training_data, input_length=3)
data_y_nino = index_calculation.get_nino34_from_data_y(data_y)
print(data_y_nino)
print(index_calculation.get_nino34_from_data(3, 461))