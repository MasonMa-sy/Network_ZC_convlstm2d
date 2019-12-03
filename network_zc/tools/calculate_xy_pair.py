file_num = 417
month = 47
# file_num = 340
# month = 124
interval = 1
prediction_month = 3
time_step = 3
# directly_months = [3, 6, 9, 12]
directly_months = [1, 2, 3, 4]

history_data = []
for i in range(file_num+time_step, file_num + month + 1):
    history_data.append(i)
print('history data:', history_data)

predict_data = []
for di, directly_month in enumerate(directly_months):
    predict_data_temp = []
    for start_month in range(file_num - prediction_month * directly_month,
                             file_num + month - prediction_month * directly_month + 1 - time_step, interval):
        predict_data_temp.append(start_month+time_step+prediction_month*directly_month)
    predict_data.append(predict_data_temp[:])
print('predict_data:', predict_data)
