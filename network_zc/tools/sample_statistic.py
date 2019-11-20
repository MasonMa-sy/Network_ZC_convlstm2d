from network_zc.model_trainer.dense_trainer_sstaha_3 import get_testing_index
from network_zc.tools import file_helper_unformatted, data_preprocess
import numpy as np

training_start = 60
training_num = 12060

# training_data, testing_data = file_helper.load_sstha_for_conv2d(training_num-1)
# data_y = np.reshape(training_data[get_testing_index(training_num), :, :, 0], (training_num2, 540))
# data_mean = np.mean(data_y, axis=0)
# file_helper.write_data(0, data_mean)

training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)
# data_y = np.reshape(training_data[:, :, :, 0], (training_num, 540))
# data_mean = np.mean(training_data, axis=0)
# data_std = np.std(training_data, axis=0)
# file_helper_unformatted.write_data(1, data_mean)
# file_helper_unformatted.write_data(2, data_std)
data_preprocess.no_month_mean(training_data, 2)
