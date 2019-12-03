"""
This script is for some parameter set.
"""
# for model
# model_name = 'convlstm_model_scnino_zc_1@test2'
model_name = 'convlstm_model_scnino_hist_1@test1'
model_type = 'conv'
# for retrain model, because ZC data and observation data are symmetrical with the equator
is_retrain = False
retrain_model_name = 'convlstm_model_scnino_zc_to_hist_1@test'
# for continue model
continue_model_name = 'conv_model_dimensionless_1_historical_sc_small@best'
# for ssim
kernel_size = 7
max_value = 10
# for seasonal circle input
is_seasonal_circle = True
# for nino index output
is_nino_output = True
# for time sequence
is_sequence = True
time_step = 3 if is_sequence else 0
# for prediction month
prediction_month = 1
# for final essay
is_best = True
predict_file_dir = 'D:\\files\课题组\zc_network\zc_data\data_networks\\'

# for reading data
# data_file = 'data_nature2'
# data_name = '\data_'
data_file = 'data_historical'
data_name = '\data_historical_'
# data_file = 'data_test'
# data_name = '\predict_data_'
# data_file = 'data_predict_0'
# data_name = '\data_'
# for data preprocess
data_file_statistics = '..\data\predict_data_'
data_preprocessing_file = '\mean\\'
# for historical_data_interpolate
data_historical_file = '\historical_data\\'
historical_binary_file = 'data_historical\data_wind'


# for data preprocess
data_preprocess_method = 'dimensionless'
