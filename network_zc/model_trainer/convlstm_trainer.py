"""
To train neural networks for zc model using convolution layers.
Add a sinusoid with the period of a year as attribute to contain information about the seasonal cycle.
Add a lambda layer to calculate nino index as a output.

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            (None, 1)            0
__________________________________________________________________________________________________
input_1 (InputLayer)            (None, 3, 20, 27, 2) 0
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 25920)     311040      input_2[0][0]
__________________________________________________________________________________________________
conv_lst_m2d_1 (ConvLSTM2D)     (None, 3, 20, 27, 16 10432       input_1[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 3, 20, 27, 16 0           embedding_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3, 20, 27, 32 0           conv_lst_m2d_1[0][0]
                                                                 reshape_1[0][0]
__________________________________________________________________________________________________
conv_lst_m2d_2 (ConvLSTM2D)     (None, 3, 20, 27, 32 73856       concatenate_1[0][0]
__________________________________________________________________________________________________
conv_lst_m2d_3 (ConvLSTM2D)     (None, 3, 20, 27, 32 73856       conv_lst_m2d_2[0][0]
__________________________________________________________________________________________________
conv_lst_m2d_4 (ConvLSTM2D)     (None, 20, 27, 2)    2456        conv_lst_m2d_3[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           conv_lst_m2d_4[0][0]
==================================================================================================
Total params: 471,640
Trainable params: 471,640
Non-trainable params: 0
__________________________________________________________________________________________________

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

from network_zc.keras_contrib.layers import NinoLayer
from network_zc.keras_contrib.losses import DSSIMObjective
import numpy as np
import time

# My libraries
from network_zc.tools import file_helper_unformatted, name_list, data_preprocess, index_calculation

# some initial parameter
model_name = name_list.model_name
data_preprocess_method = name_list.data_preprocess_method

if name_list.data_file == 'data_historical':
    training_start = 0
    training_num = 464
elif name_list.data_file == 'data_nature2':
    training_start = 60
    training_num = 12000
testing_num = 0
epochs = 2000
batch_size = 32
time_step = name_list.time_step
prediction_month = name_list.prediction_month


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# def nino_mse(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    start = time.time()
    # To define the model
    lrelu = lambda x: LeakyReLU(alpha=0.3)(x)
    inputs = Input(shape=(time_step, 20, 27, 2))
    layer1 = ConvLSTM2D(filters=16, kernel_size=3, strides=1, padding='same', return_sequences=True,
                        activation=lrelu, dropout=0.2)(inputs)
    # layer1 = BatchNormalization()(layer1)

    sc_input = Input(shape=(1,), dtype='int32')
    sc_layer = Embedding(12, 3 * 20 * 27 * 16, input_length=1)(sc_input)
    # sc_layer = Flatten()(sc_layer)
    # sc_layer = Dense(3*20*27*64)(sc_layer)
    # # sc_layer = BatchNormalization()(sc_layer)  # ?
    # sc_layer = LeakyReLU(alpha=0.3)(sc_layer)  # ?
    # sc_layer = Dropout(0.2)(sc_layer)
    sc_layer = Reshape((3, 20, 27, 16))(sc_layer)
    layer2 = concatenate([layer1, sc_layer], axis=4)

    layer2 = ConvLSTM2D(filters=32, kernel_size=3, strides=1, padding='same', return_sequences=True,
                        activation=lrelu, dropout=0.2)(layer2)
    # print(layer1.get_shape().as_list())
    # layer2 = BatchNormalization()(layer2)
    layer3 = ConvLSTM2D(filters=32, kernel_size=3, strides=1, padding='same', return_sequences=True,
                        activation=lrelu, dropout=0.2)(layer2)
    # layer3 = BatchNormalization()(layer3)
    predictions = ConvLSTM2D(filters=2, padding='same', kernel_size=3, strides=1, activation='linear')(layer3)
    nino_layer = NinoLayer.get_nino_layer()
    predictions_ninos = nino_layer(predictions)

    model = Model(inputs=[inputs, sc_input], outputs=[predictions, predictions_ninos])
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, loss=mean_squared_error, loss_weights=[0.25, 0.75],
                  metrics=[root_mean_squared_error])
    model.summary()
    # to train model
    # the data for training is ssta and ha
    training_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, training_num)
    training_data = data_preprocess.data_preprocess(training_data, 0, data_preprocess_method)
    data_x, data_y = data_preprocess.sequence_data(training_data, input_length=time_step,
                                                   prediction_month=prediction_month)
    data_y_nino = index_calculation.get_nino34_from_data_y(data_y)
    # print(data_y_nino)
    sc = np.linspace(0, 11, 12, dtype='int32')
    sc = np.tile(sc, int((training_num - training_start) / 12 + 1))
    data_sc = sc[:(training_num - training_start - time_step + 1 - prediction_month + 1)]
    # print(data_x.shape, data_y.shape)

    # tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    save_best = ModelCheckpoint('..\..\model\\best\\' + model_name + '.h5',
                                monitor='val_loss',
                                verbose=1, save_best_only=True, mode='min', period=1)
    train_hist = model.fit([data_x, data_sc], [data_y, data_y_nino], batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[save_best], validation_split=0.1)

    # To save the model and logs
    # model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper_unformatted.find_logs_final(model_name + '_train'), 'w') as f:
        f.write(str(train_hist.history))
        f.write(str(save_best.best))
    print(time.time() - start)
