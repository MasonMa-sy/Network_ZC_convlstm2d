"""
For final paper plot and analysis
According to model_name and directly_month.
plot and calculate correlation coefficient (overlap part).
"""
# Third-party libraries
import h5py
from keras.layers import Input, Dense
from keras import optimizers
from keras.models import Model
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
# My libraries
from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.plot_script.figure_plot import predict_369_data
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation, math_tool
from network_zc.model_trainer import dense_trainer
from network_zc.model_trainer import dense_trainer_sstaha
from network_zc.model_trainer import dense_trainer_sstaha_4
from network_zc.model_trainer import convolutional_trainer_alex


model_name = name_list.model_name
if name_list.is_best:
    model_name = file_helper_unformatted.find_model_best(model_name)
model_type = name_list.model_type
is_retrain = name_list.is_retrain
is_seasonal_circle = name_list.is_seasonal_circle

"""
file_num: the first prediction of data num
month: month num to predict
interval: Prediction interval
prediction_month: For the model to predict month num
directly_month: rolling run model times
"""
file_num = predict_369_data.file_num
month = predict_369_data.month
interval = 1
prediction_month = predict_369_data.prediction_month
directly_month = predict_369_data.directly_month
predict_file_dir = name_list.predict_file_dir

# make directory
file_path = predict_file_dir + model_name + '\\' + str(directly_month) + '\\'

data_x = np.empty([1, 20, 27, 2])

x = np.linspace(file_num, file_num+month, month+1)
nino34_from_data = index_calculation.get_nino34_from_data(file_num, month)
plt.plot(x, nino34_from_data, 'r', linewidth=1)
plt.tick_params(labelsize=15)

f = h5py.File(file_path+'nino34.h5', 'r')
X = f['X']
Y = f['Y']

# plt.legend(['prediction', 'ZCdata'], loc='upper right')
plt.plot(X, Y, 'y', linewidth=1, linestyle='--')
print(model_name)
print(math_tool.pearson_distance(Y[:month-directly_month*prediction_month],
                                 nino34_from_data[directly_month*prediction_month:]))
f.close()
plt.grid(True, linestyle='--', color='gray', linewidth='1', alpha=0.5)
plt.show()
