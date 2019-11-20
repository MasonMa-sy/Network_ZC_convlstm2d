"""
For final paper plot and analysis
According to directly_month.
Calculate the RMSE of ZC data.
"""
# Third-party libraries

# My libraries

from network_zc.plot_script.figure_plot import predict_369_data
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation, math_tool
import matplotlib.pyplot as plt

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
file_num = 0
month = 265
interval = 1
prediction_month = 12
directly_month = predict_369_data.directly_month
data_preprocess_method = name_list.data_preprocess_method

# make directory
file_path = "D:\msy\projects\zc\zcdata\data_predict_" + str(prediction_month)
file_hisotircal = 'D:\msy\projects\zc\zcdata\data_historical'

rmse_all = []
nino34 = []
nino34_historical = []
for start_month in range(file_num+prediction_month, file_num+month+prediction_month+1, interval):
    data_x = file_helper_unformatted.read_data_best(file_path, start_month+192)
    data_from_data = file_helper_unformatted.read_data_historical(file_hisotircal, start_month)
    nino34_temp = index_calculation.get_nino34(data_x)
    nino34.append(nino34_temp)
    nino34_temp = index_calculation.get_nino34(data_from_data)
    nino34_historical.append(nino34_temp)
    rmse_all.append(math_tool.calculate_rmse(data_x[:, :, 0], data_from_data[:, :, 0]))

print(math_tool.pearson_distance(nino34, nino34_historical))
plt.plot(nino34, 'r', linewidth=1)
plt.plot(nino34_historical, 'y', linewidth=1)
print(rmse_all)
print(sum(rmse_all)/rmse_all.__len__())
plt.show()
