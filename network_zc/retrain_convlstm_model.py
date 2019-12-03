"""

"""
# Third-party libraries
from keras.layers import Input, Dense, LeakyReLU
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
# My libraries
from network_zc.keras_contrib.losses import DSSIMObjective, DSSIMObjectiveCustom
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list
from network_zc.tools import index_calculation
from network_zc.model_trainer import dense_trainer
from network_zc.model_trainer import dense_trainer_sstaha
from network_zc.model_trainer import dense_trainer_sstaha_4
from network_zc.model_trainer import convolutional_trainer_alex


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    # Load the model
    model_name = name_list.model_name
    retrain_model_name = name_list.retrain_model_name
    model_type = name_list.model_type
    is_retrain = name_list.is_retrain
    is_seasonal_circle = name_list.is_seasonal_circle
    time_step = name_list.time_step
    prediction_month = name_list.prediction_month
    # Load the model
    if model_type == 'conv':
        lrelu = lambda y: LeakyReLU(alpha=0.3)(y)
        model = load_model('..\model\\best\\' + model_name + '.h5',
                           custom_objects={'mean_squared_error': mean_squared_error,
                                           'root_mean_squared_error': root_mean_squared_error,
                                           'mean_absolute_error': mean_absolute_error,
                                           '<lambda>': lrelu})
    elif model_type == 'dense':
        model = load_model('..\model\\best\\' + model_name + '.h5',
                           custom_objects={'mean_squared_error': mean_squared_error
                               , 'root_mean_squared_error': root_mean_squared_error,
                                           'mean_absolute_error': mean_absolute_error})

    #
    print(len(model.layers))
    training_start = 0
    training_num = 464
    batch_size = 32
    epochs = 2000

    #
    data_preprocess_method = name_list.data_preprocess_method
    all_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)
    all_data = data_preprocess.data_preprocess(all_data, 0, data_preprocess_method)
    data_x, data_y = data_preprocess.sequence_data(all_data, input_length=time_step,
                                                   prediction_month=prediction_month)
    if is_retrain:
        data_x = file_helper_unformatted.exchange_rows(data_x)
    data_y_nino = index_calculation.get_nino34_from_data_y(data_y)
    # print(data_y_nino)
    sc = np.linspace(0, 11, 12, dtype='int32')
    sc = np.tile(sc, int((training_num - training_start) / 12 + 1))
    data_sc = sc[:(training_num - training_start - time_step + 1 - prediction_month + 1)]
    # all_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, all_num)

    # for layer in model.layers[:]:
    #     layer.trainable = False
    # for layer in model.layers[6:20]:
    #     layer.trainable = True
    for layer in model.layers:
        print(layer.trainable)
    adam = optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # adam = optimizers.Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    if model_type == 'dense':
        model.compile(optimizer=adam, loss=mean_squared_error, metrics=[mean_squared_error, root_mean_squared_error
            , mean_absolute_error])
    elif model_type == 'conv':
        model.compile(optimizer=adam, loss=mean_squared_error,
                      metrics=[root_mean_squared_error, mean_absolute_error, mean_squared_error])
    # tesorboard = TensorBoard('..\model\\tensorboard\\' + retrain_model_name)
    save_best = ModelCheckpoint('..\model\\best\\' + retrain_model_name + '.h5',
                                monitor='val_loss',
                                verbose=1, save_best_only=True, mode='min', period=1)

    train_hist = model.fit([data_x, data_sc], [data_y, data_y_nino], batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[save_best], validation_split=0.1)
    # model.save('..\model\\' + retrain_model_name + '.h5')
    with open('..\model\\best\\logs\\' + retrain_model_name + '_train', 'w') as f:
        f.write(str(train_hist.history))
        f.write(str(save_best.best))
