"""
To train neural networks for zc model using convolution layers.
Add a sinusoid with the period of a year as attribute to contain information about the seasonal cycle.
"""
# Third-party libraries
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, concatenate, \
    Embedding
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
import keras.backend as K
from network_zc.keras_contrib.losses import DSSIMObjective
import numpy as np

# My libraries
from network_zc.keras_contrib.losses.dssim_custom import DSSIMObjectiveCustom
from network_zc.tools import file_helper_unformatted, name_list, data_preprocess

# some initial parameter
model_name = name_list.model_name
data_preprocess_method = name_list.data_preprocess_method
is_best = name_list.is_best

training_start = 0
training_num = 464
testing_num = 0
epochs = 200
batch_size = 32
kernel_size = name_list.kernel_size
max_value = name_list.max_value


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    # To define the model
    inputs = Input(shape=(20, 27, 2))                                                   # 0
    #layer1 = BatchNormalization()(inputs)                                               # 1
    # 20x27 to 9x12
    layer1 = Conv2D(filters=32, kernel_size=(4, 5), strides=2, padding='valid')(inputs)     # 2
    #layer1 = BatchNormalization()(layer1)                                               # 3
    layer1 = LeakyReLU(alpha=0.3)(layer1)                                               # 4
    layer1 = Dropout(0.2)(layer1)                                                       # 5
    # 9x12 to 7x10
    layer2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid')(layer1)      # 6
    #layer2 = BatchNormalization()(layer2)                                               # 7
    layer2 = LeakyReLU(alpha=0.3)(layer2)                                               # 8
    layer2 = Dropout(0.2)(layer2)                                                       # 9
    # print(layer2.get_shape().as_list())
    layer3 = Flatten()(layer2)                                                          # 10
    # add sc input
    sc_input = Input(shape=(1,), dtype='int32')
    sc_layer = Embedding(12, 64, input_length=1)(sc_input)
    sc_layer = Flatten()(sc_layer)
    sc_layer = Dense(70*64)(sc_layer)
    #sc_layer = BatchNormalization()(sc_layer)  # ?
    sc_layer = LeakyReLU(alpha=0.3)(sc_layer)  # ?
    sc_layer = Dropout(0.2)(sc_layer)
    layer3 = concatenate([layer3, sc_layer])
    layer3 = Dense(7*10*64)(layer3)                                                     # 11
    #layer3 = BatchNormalization()(layer3)                                               # 12
    layer3 = LeakyReLU(alpha=0.3)(layer3)                                               # 13
    layer3 = Dropout(0.2)(layer3)                                                       # 14
    layer3 = Reshape((7, 10, 64))(layer3)                                               # 15
    layer4 = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='valid')(layer3)     # 16
    #layer4 = BatchNormalization()(layer4)                                               # 17
    layer4 = LeakyReLU(alpha=0.3)(layer4)                                               # 18
    layer4 = Dropout(0.2)(layer4)                                                       # 19
    predictions = Conv2DTranspose(filters=2, kernel_size=(4, 5), strides=2,
                                  padding='valid', activation='linear')(layer4)         # 20

    model = Model(inputs=[inputs, sc_input], outputs=predictions)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    ssim = DSSIMObjectiveCustom(kernel_size=kernel_size, max_value=max_value)
    ssim_metrics = DSSIMObjectiveCustom(kernel_size=7, max_value=10)

    def ssim_l1(y_true, y_pred):
        a = 0.84
        return a*ssim(y_true, y_pred) + (1-a)*mean_absolute_error(y_true, y_pred)

    model.compile(optimizer=adam, loss=mean_squared_error,
                  metrics=[root_mean_squared_error, ssim_metrics, mean_absolute_error, mean_squared_error])
    # to train model
    # the data for training is ssta and ha
    training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)

    # data preprocess z-zero
    if data_preprocess_method == 'preprocess_Z':
        training_data = data_preprocess.preprocess_Z(training_data, 0)
    # data preprocess dimensionless
    if data_preprocess_method == 'dimensionless':
        training_data = data_preprocess.dimensionless(training_data, 0)
    # data preprocess 0-1
    if data_preprocess_method == 'preprocess_01':
        training_data = data_preprocess.preprocess_01(training_data, 0)
    # data preprocess no month mean
    if data_preprocess_method == 'nomonthmean':
        training_data = data_preprocess.no_month_mean(training_data, 0)

    data_x = training_data[:-1]
    data_y = training_data[1:]
    sc = np.linspace(0, 11, 12, dtype='int32')
    sc = np.tile(sc, int((training_num-training_start)/12+1))
    data_sc = sc[:(training_num-training_start)]
    # tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    if is_best:
        save_best = ModelCheckpoint('..\..\model\\best\\' + model_name + '@' + 'best.h5',
                                monitor='val_root_mean_squared_error',
                                verbose=1, save_best_only=True, mode='min', period=1)
    else:
        save_best = ModelCheckpoint('..\..\model\\' + model_name + '.h5',
                                monitor='val_root_mean_squared_error',
                                verbose=1, save_best_only=True, mode='min', period=1)
    train_hist = model.fit([data_x, data_sc], data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[save_best], validation_split=0.2)

    # To save the model and logs
    # model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper_unformatted.find_logs_final(model_name+'_train'), 'w') as f:
        f.write(str(train_hist.history))
        f.write(str(save_best.best))
