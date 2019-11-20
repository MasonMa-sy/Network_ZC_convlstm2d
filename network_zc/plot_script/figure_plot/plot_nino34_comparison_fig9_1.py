"""
For final paper plot and analysis
According to model_name and directly_month.
plot and nino3.4 .
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
from matplotlib.legend import Legend

from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.plot_script.figure_plot import predict_369_data
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation, math_tool
from network_zc.model_trainer import dense_trainer
from network_zc.model_trainer import dense_trainer_sstaha
from network_zc.model_trainer import dense_trainer_sstaha_4
from network_zc.model_trainer import convolutional_trainer_alex


model_name1 = 'conv_model_dimensionless_1_ZC_sc_1@best'
if name_list.is_best:
    model_name1 = file_helper_unformatted.find_model_best(model_name1)
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
# file_num = 10800
# month = 100
file_num = 0
month = 265
interval = 1
prediction_month = 12
directly_month = predict_369_data.directly_month
data_preprocess_method = name_list.data_preprocess_method

# make directory
file_path0 = "D:\msy\projects\zc\zcdata\data_predict_" + str(0)
file_path1 = "D:\msy\projects\zc\zcdata\data_predict_" + str(3)
file_path2 = "D:\msy\projects\zc\zcdata\data_predict_" + str(6)
file_path3 = "D:\msy\projects\zc\zcdata\data_predict_" + str(9)
file_path4 = "D:\msy\projects\zc\zcdata\data_predict_" + str(12)
file_hisotircal = 'D:\msy\projects\zc\zcdata\data_historical'

data_x = np.empty([1, 20, 27, 2])

fig, ax = plt.subplots()
x = np.linspace(file_num, file_num+month+12, month+1+12)
nino34_from_data = index_calculation.get_nino34_from_data(file_num, month+12)
line1, = plt.plot(x, nino34_from_data, 'black', linewidth=2.5)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

nino34 = []
for start_month in range(file_num+0, file_num+month+0+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path0, start_month+192)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    X = np.linspace(file_num+0, file_num+month+0, month+1)
# plt.legend(['prediction', 'ZCdata'], loc='upper right')
line2, = plt.plot(X, nino34, 'b', linewidth=1.5, linestyle='-')

nino34 = []
for start_month in range(file_num+3, file_num+month+3+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path1, start_month+192)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    X = np.linspace(file_num+3, file_num+month+3, month+1)
line3, = plt.plot(X, nino34, 'r', linewidth=1.5, linestyle='-')

nino34 = []
for start_month in range(file_num+6, file_num+month+6+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path2, start_month+192)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    X = np.linspace(file_num+6, file_num+month+6, month+1)
line4, = plt.plot(X, nino34, 'y', linewidth=1.5, linestyle='-')

nino34 = []
for start_month in range(file_num+9, file_num+month+9+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path3, start_month+192)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    X = np.linspace(file_num+9, file_num+month+9, month+1)
line5, = plt.plot(X, nino34, 'g', linewidth=1.5, linestyle='-')

nino34 = []
for start_month in range(file_num+12, file_num+month+12+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path4, start_month+192)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    X = np.linspace(file_num+12, file_num+month+12, month+1)
line6, = plt.plot(X, nino34, 'm', linewidth=1.5, linestyle='-')

last_month = file_num+month+12
x_ticks1 = np.arange(0, last_month, 24, dtype=int)
x_ticks2 = np.arange(80, 105, 2, dtype=int)
x_ticks2 = np.where(x_ticks2 < 100, x_ticks2, x_ticks2-100)
x_ticks2 = ['Jan-'+str(x) for x in x_ticks2]
plt.xticks(x_ticks1, x_ticks2)

font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15}
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10}
plt.grid(True, linestyle='--', color='gray', linewidth='1', alpha=0.5)
row1 = 'historical'
row2 = '0-month'
row3 = '3-month'
row4 = '6-month'
row5 = '9-month'
row6 = '12-month'

plt.legend([row1, row2, row3, row4, row5, row6],
           loc='upper right', prop=font2, framealpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('ZC Predictions with Wind Stress Forced', fontsize=15, fontname='Times New Roman')
ax.set_ylabel('NINO3.4 SSTA(°C)', font1)
ax.set_xlabel('Time(month)', font1)
plt.show()
