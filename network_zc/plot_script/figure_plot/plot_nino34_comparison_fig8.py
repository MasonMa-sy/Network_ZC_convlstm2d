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


model_name1 = 'conv_model_dimensionless_1_ZC_sc_1_to_historical@best'
model_name2 = 'conv_model_dimensionless_1_historical_sc_small_2@best'
model_name3 = 'conv_model_dimensionless_1_historical_sc_5@best'
if name_list.is_best:
    model_name1 = file_helper_unformatted.find_model_best(model_name1)
    model_name2 = file_helper_unformatted.find_model_best(model_name2)
    model_name3 = file_helper_unformatted.find_model_best(model_name3)
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
file_num = 417
month = 47
interval = 1
prediction_month = 1
directly_month = 12

# make directory
file_path1 = "D:\msy\projects\zc\zcdata\data_networks\\" + model_name1 + '\\' + str(directly_month) + '\\'
file_path2 = "D:\msy\projects\zc\zcdata\data_networks\\" + model_name2 + '\\' + str(directly_month) + '\\'
file_path3 = "D:\msy\projects\zc\zcdata\data_networks\\" + model_name3 + '\\' + str(directly_month) + '\\'

data_x = np.empty([1, 20, 27, 2])

fig, ax = plt.subplots()
x = np.linspace(file_num, file_num+month, month+1)
nino34_from_data = index_calculation.get_nino34_from_data(file_num, month)
line0 = plt.plot(x, nino34_from_data, 'b', linewidth=0, linestyle='-')
line1, = plt.plot(x, nino34_from_data, 'black', linewidth=1.5)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

f = h5py.File(file_path1+'nino34.h5', 'r')
X = f['X']
Y = f['Y']
# plt.legend(['prediction', 'ZCdata'], loc='upper right')
line2, = plt.plot(X, Y, 'b', linewidth=1.5, linestyle='-')
print(model_name1)
print(math_tool.pearson_distance(Y[:month-directly_month*prediction_month],
                                 nino34_from_data[directly_month*prediction_month:]))
f.close()

f = h5py.File(file_path2+'nino34.h5', 'r')
X = f['X']
Y = f['Y']
line3, = plt.plot(X, Y, 'r', linewidth=1.5, linestyle='-')
print(model_name2)
print(math_tool.pearson_distance(Y[:month-directly_month*prediction_month],
                                 nino34_from_data[directly_month*prediction_month:]))
f.close()

f = h5py.File(file_path3+'nino34.h5', 'r')
X = f['X']
Y = f['Y']
line4, = plt.plot(X, Y, 'y', linewidth=1.5, linestyle='-')
print(model_name3)
print(math_tool.pearson_distance(Y[:month-directly_month*prediction_month],
                                 nino34_from_data[directly_month*prediction_month:]))
last_month = X[-1]
f.close()

x_ticks1 = np.arange(420, last_month, 12, dtype=int)
x_ticks2 = ['Jan-'+str(x) for x in np.arange(15, 20, 1, dtype=int)]
plt.xticks(x_ticks1, x_ticks2)

font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15}
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 11.5}
plt.grid(True, linestyle='--', color='gray', linewidth='1', alpha=0.5)
row1 = '                         Correlation  RMSE'
row2 = 'historical'
if directly_month == 3:
    row3 = 'CNN-TRANS 0.96821         0.37360'
    row4 = 'CNN-HIS-S    0.95731         0.37819'
    row5 = 'CNN-HIS       0.96365         0.39543'
if directly_month == 6:
    row3 = 'CNN-TRANS 0.91123         0.46731'
    row4 = 'CNN-HIS-S    0.85527         0.48672'
    row5 = 'CNN-HIS       0.87537         0.49003'
if directly_month == 9:
    row3 = 'CNN-TRANS 0.90546         0.52458'
    row4 = 'CNN-HIS-S    0.75553         0.55892'
    row5 = 'CNN-HIS       0.71461         0.54648'
if directly_month == 12:
    row3 = 'CNN-TRANS 0.90815         0.51826'
    row4 = 'CNN-HIS-S    0.76564         0.54631'
    row5 = 'CNN-HIS       0.71146         0.52277'
plt.legend([row1, row2, row3, row4, row5],
           loc='upper right', prop=font2, framealpha=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title(str(directly_month)+'-month lead time', fontsize=15, fontname='Times New Roman')
ax.set_ylabel('NINO3.4 SSTA(°C)', font1)
ax.set_xlabel('Time(month)', font1)
plt.show()
