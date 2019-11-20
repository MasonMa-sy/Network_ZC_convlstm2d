"""

"""
# Third-party libraries
from keras.layers import Input, Dense
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
    continue_model_name = name_list.continue_model_name
    model_type = name_list.model_type
    is_retrain = name_list.is_retrain
    is_seasonal_circle = name_list.is_seasonal_circle
    # Load the model
    if model_type == 'conv':
        kernel_size = name_list.kernel_size
        max_value = name_list.max_value
        ssim = DSSIMObjectiveCustom(kernel_size=kernel_size, max_value=max_value)
        ssim_metrics = DSSIMObjectiveCustom(kernel_size=7, max_value=10)

        def ssim_l1(y_true, y_pred):
            a = 0.84
            return a * ssim(y_true, y_pred) + (1 - a) * mean_absolute_error(y_true, y_pred)

        model = load_model('..\model\\best\\' + continue_model_name + '@best.h5', custom_objects={'mean_squared_error': mean_squared_error,
            'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error,
            'ssim_metrics': ssim_metrics, 'ssim_l1': ssim_l1, 'DSSIMObjective': ssim})
    elif model_type == 'dense':
        model = load_model('..\model\\best\\' + continue_model_name + '@best.h5', custom_objects={'mean_squared_error': mean_squared_error
            , 'root_mean_squared_error': root_mean_squared_error, 'mean_absolute_error': mean_absolute_error})
    print(len(model.layers))
    training_start = 0
    all_num = 464
    batch_size = 32
    epochs = 200
    data_preprocess_method = name_list.data_preprocess_method

    all_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, all_num)
    if is_retrain:
        all_data = file_helper_unformatted.exchange_rows(all_data)
    if is_seasonal_circle:
        sc = np.linspace(0, 11, 12, dtype='int32')
        sc = np.tile(sc, int((all_num - training_start) / 12 + 1))
        data_sc = sc[:(all_num - training_start)]

    # data preprocess z-zero
    if data_preprocess_method == 'preprocess_Z':
        all_data = data_preprocess.preprocess_Z(all_data, 0)
    # data preprocess dimensionless
    if data_preprocess_method == 'dimensionless':
        all_data = data_preprocess.dimensionless(all_data, 0)
    # data preprocess 0-1
    if data_preprocess_method == 'preprocess_01':
        all_data = data_preprocess.preprocess_01(all_data, 0)
    # data preprocess no month mean
    if data_preprocess_method == 'nomonthmean':
        all_data = data_preprocess.no_month_mean(all_data, 0)

    if model_type == 'conv':
        data_x = all_data[:-1]
        data_y = all_data[1:]
    elif model_type == 'dense':
        data_x = np.reshape(all_data[:-1], (all_num, 1080))
        data_y = np.reshape(all_data[1:], (all_num, 1080))

    # for layer in model.layers[:]:
    #     layer.trainable = False
    # for layer in model.layers[6:20]:
    #     layer.trainable = True
    for layer in model.layers:
        print(layer.trainable)
    save_best = ModelCheckpoint('..\model\\best\\' + continue_model_name + '@' + 'best.h5',
                                monitor='val_root_mean_squared_error',
                                verbose=1, save_best_only=True, mode='min', period=1)
    if is_seasonal_circle:
        train_hist = model.fit([data_x, data_sc], data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                               callbacks=[save_best], validation_split=0.1)
    else:
        train_hist = model.fit(data_x, data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                               callbacks=[save_best], validation_split=0.1)
    # model.save('..\model\\' + continue_model_name + '.h5')
    with open('..\model\\best\\logs\\'+continue_model_name+'_train', 'w') as f:
        f.write(str(train_hist.history))
        f.write(str(save_best.best))
