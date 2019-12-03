from sklearn.model_selection import RepeatedKFold, KFold
import numpy as np
from keras.activations import relu
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, concatenate, \
    Embedding, ConvLSTM2D, BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
import time
import json
import gc

# My libraries
from network_zc.tools import file_helper_unformatted, name_list, data_preprocess, index_calculation, json_encoder
from network_zc.model import custom_function, convlstm

if __name__ == '__main__':
    start = time.time()
    model_name = name_list.model_name
    data_preprocess_method = name_list.data_preprocess_method
    is_seasonal_circle = name_list.is_seasonal_circle
    is_nino_output = name_list.is_nino_output

    if name_list.data_file == 'data_historical':
        training_start = 0
        training_num = 464
    elif name_list.data_file == 'data_nature2':
        training_start = 60
        training_num = 12000
    testing_num = 0
    epochs = 1
    batch_size = 32
    time_step = name_list.time_step
    prediction_month = name_list.prediction_month

    training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)
    training_data = data_preprocess.data_preprocess(training_data, 0, data_preprocess_method)
    data_x, data_y = data_preprocess.sequence_data(training_data, input_length=time_step,
                                                   prediction_month=prediction_month)
    if is_nino_output:
        data_y_nino = index_calculation.get_nino34_from_data_y(data_y)
    # print(data_y_nino)
    if is_seasonal_circle:
        sc = np.linspace(0, 11, 12, dtype='int32')
        sc = np.tile(sc, int((training_num - training_start) / 12 + 1))
        data_sc = sc[:(training_num - training_start - time_step + 1 - prediction_month + 1)]
    # print(data_x.shape, data_y.shape)

    model_num = 0
    models_summary = {}
    kf = KFold(n_splits=10, random_state=0)
    for train_index, test_index in kf.split(data_x):
        # print('train_index', train_index, 'test_index', test_index)

        model = convlstm.ConvlstmModel(model_name, time_step, prediction_month, is_seasonal_circle,
                                       is_nino_output).get_model()
        save_best = ModelCheckpoint('..\model\\cross_validation\\' + model_name + ' ' + str(test_index[0]) + '.h5',
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True, mode='min', period=1)
        train_x, train_y = data_x[train_index], data_y[train_index]
        test_x, test_y = data_x[test_index], data_y[test_index]
        if is_seasonal_circle:
            train_sc, test_sc = data_sc[train_index], data_sc[test_index]
        if is_nino_output:
            train_nino, test_nino = data_y_nino[train_index], data_y_nino[test_index]
        if not is_seasonal_circle and not is_nino_output:
            train_hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
                                   verbose=2,
                                   callbacks=[save_best],
                                   validation_data=(test_x, test_y))
        elif is_seasonal_circle and not is_nino_output:
            train_hist = model.fit([train_x, train_sc], train_y, batch_size=batch_size, epochs=epochs,
                                   verbose=2,
                                   callbacks=[save_best],
                                   validation_data=([test_x, test_sc], test_y))
        elif not is_seasonal_circle and is_nino_output:
            train_hist = model.fit(train_x, [train_y, train_nino], batch_size=batch_size, epochs=epochs,
                                   verbose=2,
                                   callbacks=[save_best],
                                   validation_data=(test_x, [test_y, test_nino]))
        else:
            train_hist = model.fit([train_x, train_sc], [train_y, train_nino], batch_size=batch_size, epochs=epochs,
                                   verbose=2,
                                   callbacks=[save_best],
                                   validation_data=([test_x, test_sc], [test_y, test_nino]))

        # To save the model and logs
        # model.save('..\..\model\\' + model_name + '.h5')
        with open(file_helper_unformatted.find_logs_cross(model_name + ' ' + str(test_index[0]) + '_train'), 'w') as f:
            f.write(str(train_hist.history))
            f.write(str(save_best.best))
        del model, train_index, train_y, test_x, test_y
        if is_seasonal_circle:
            del train_sc, test_sc
        if is_nino_output:
            del train_nino, test_nino
        K.clear_session()
        gc.collect()
        model_num += 1
        hist_index = train_hist.history['val_loss'].index(save_best.best)
        for k, v in train_hist.history.items():
            models_summary.setdefault(k, []).append(v[hist_index])
        print(model_num, ' finished ', test_index[0])
    file_path = '..\model\\cross_validation\\logs\\' + model_name
    with open(file_path + 's.log', 'w') as f:
        res2 = json.dumps(models_summary, cls=json_encoder.JsonEncoder, indent=4)
        f.write(res2)
    print(time.time() - start)
