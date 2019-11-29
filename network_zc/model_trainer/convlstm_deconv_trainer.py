"""
To train neural networks for zc model using convolution layers.
Add a sinusoid with the period of a year as attribute to contain information about the seasonal cycle.
"""
# Third-party libraries
from keras.activations import relu
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, concatenate, \
    Embedding, ConvLSTM2D, BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
from network_zc.keras_contrib.losses import DSSIMObjective
import numpy as np
import time

# My libraries
from network_zc.tools import file_helper_unformatted, name_list, data_preprocess

# some initial parameter
model_name = name_list.model_name
data_preprocess_method = name_list.data_preprocess_method

training_start = 0
training_num = 464
testing_num = 0
epochs = 500
batch_size = 32
time_step = name_list.time_step


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    start = time.time()
    # To define the model
    lrelu = lambda x: LeakyReLU(alpha=0.3)(x)
    inputs = Input(shape=(time_step, 20, 27, 2))
    layer1 = ConvLSTM2D(filters=32, kernel_size=(4, 5), strides=2, padding='valid', return_sequences=True,
                        activation=lrelu, dropout=0.2)(inputs)
    # layer1 = BatchNormalization()(layer1)
    layer2 = ConvLSTM2D(filters=64, kernel_size=3, strides=1, padding='valid', return_sequences=False,
                        activation=lrelu, dropout=0.2)(layer1)
    # layer2 = BatchNormalization()(layer2)
    # layer3 = ConvLSTM2D(filters=64, kernel_size=3, strides=1, padding='same', return_sequences=False,
    #                     activation=lrelu, dropout=0.2)(layer2)
    # layer3 = BatchNormalization()(layer3)
    sc_input = Input(shape=(1,), dtype='int32')
    sc_layer = Embedding(12, 128, input_length=1)(sc_input)
    sc_layer = Flatten()(sc_layer)
    sc_layer = Dense(70 * 64)(sc_layer)
    # sc_layer = BatchNormalization()(sc_layer)  # ?
    sc_layer = LeakyReLU(alpha=0.3)(sc_layer)  # ?
    sc_layer = Dropout(0.2)(sc_layer)
    sc_layer = Reshape((7, 10, 64))(sc_layer)
    layer4 = concatenate([layer2, sc_layer], axis=3)

    layer4 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(layer4)
    layer4 = LeakyReLU(alpha=0.3)(layer4)  # 13
    # layer3 = BatchNormalization()(layer3)                                               # 17
    layer4 = Dropout(0.2)(layer4)  # 14
    # layer3 = Reshape((7, 10, 64))(layer3)                                               # 15

    layer5 = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='valid')(layer4)  # 16
    # layer4 = BatchNormalization()(layer4)                                               # 17
    layer5 = LeakyReLU(alpha=0.3)(layer5)  # 18
    layer5 = Dropout(0.2)(layer5)  # 19
    predictions = Conv2DTranspose(filters=2, kernel_size=(4, 5), strides=2,
                                  padding='valid', activation='linear')(layer5)  # 20
    model = Model(inputs=[inputs, sc_input], outputs=predictions)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, loss=mean_squared_error,
                  metrics=[root_mean_squared_error, mean_absolute_error, mean_squared_error])
    model.summary()
    # to train model
    # the data for training is ssta and ha
    training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)
    training_data = data_preprocess.data_preprocess(training_data, 0, data_preprocess_method)
    data_x, data_y = data_preprocess.sequence_data(training_data, input_length=time_step)
    sc = np.linspace(0, 11, 12, dtype='int32')
    sc = np.tile(sc, int((training_num - training_start) / 12 + 1))
    data_sc = sc[:(training_num - training_start-time_step+1)]
    print(data_x.shape, data_y.shape)

    # sc = np.linspace(0, 11, 12, dtype='int32')
    # sc = np.tile(sc, int((training_num-training_start)/12+1))
    # data_sc = sc[:(training_num-training_start)]
    # tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    save_best = ModelCheckpoint('..\..\model\\best\\' + model_name + '.h5',
                                monitor='val_root_mean_squared_error',
                                verbose=1, save_best_only=True, mode='min', period=1)
    train_hist = model.fit([data_x, data_sc], data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[save_best], validation_split=0.1)

    # To save the model and logs
    # model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper_unformatted.find_logs_final(model_name+'_train'), 'w') as f:
        f.write(str(train_hist.history))
        f.write(str(save_best.best))
    print(time.time()-start)
