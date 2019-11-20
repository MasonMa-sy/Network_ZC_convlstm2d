"""
This script is to plot the fig of loss
Add a script as Fig3, so the 'Fig4' change to 'Fig5'
"""
# Third-party libraries
import matplotlib.pyplot as plt
import json
# My libraries
from network_zc.tools import file_helper_unformatted, name_list

model_name = name_list.model_name
log_path = '..\..\..\model\\best\\logs\\'
model_name_temp = 'conv_model_dimensionless_1_historical_sc_5_train'
model_name_temp2 = 'conv_model_dimensionless_1_ZC_sc_1_to_historical_train'
if __name__ == '__main__':
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    with open(file_helper_unformatted.find_logs(log_path+model_name_temp), 'r') as file:
        data = file.read()
        history = json.loads(data.replace("'", '"'))
        # # first subplot
        # plt.subplot(2, 1, 1)
        # plt.plot(history['val_loss'])
        # plt.plot(history['loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['val', 'train'], loc='upper right')
        # second subplot
        # plt.subplot(2, 1, 2)
        plt.plot(history['val_root_mean_squared_error'], linewidth='2.5')
        plt.plot(history['root_mean_squared_error'], linewidth='2.5')
        plt.title('model rmse', font1)
        plt.ylabel('RMSE', font1)
        plt.xlabel('epoch', font1)
        # plt.legend(['val', 'train'], loc='upper right')

    with open(file_helper_unformatted.find_logs(log_path + model_name_temp2), 'r') as file:
        data2 = file.read()
        history2 = json.loads(data2.replace("'", '"'))
        # first subplot
        plt.plot(history2['val_root_mean_squared_error'], linewidth='2.5')
        plt.legend(['CNN-HIS val', 'CNN-HIS train', 'CNN-TRANS val'], loc='upper right', prop=font1)
        plt.xlim(0, 200)
    plt.hlines(0.13831, 0, 200, color='darkgrey', linestyles=':')
    plt.scatter(0, 0.17312, s=80, c='red', alpha=0.8, marker='o')
    plt.scatter(0, 0.23901, s=80, c='red', alpha=0.8, marker='o')
    plt.tick_params(labelsize=15)
    plt.show()




