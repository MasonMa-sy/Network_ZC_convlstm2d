from keras.activations import relu
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, concatenate, \
    Embedding, ConvLSTM2D, BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
import numpy as np

from network_zc.keras_contrib.layers import NinoLayer
from network_zc.keras_contrib.losses import DSSIMObjective
from network_zc.model import custom_function


class ConvlstmModel:
    def __init__(self, model_name, time_step, prediction_month, is_seasonal_circle, is_nino_output):
        self.model_name = model_name
        self.time_step = time_step
        self.prediction_month = prediction_month
        self.is_seasonal_circle = is_seasonal_circle
        self.is_nino_output = is_nino_output

    def get_model(self):
        lrelu = custom_function.lrelu()
        inputs = Input(shape=(self.time_step, 20, 27, 2), name='grid_input')
        layer1 = ConvLSTM2D(filters=16, kernel_size=3, strides=1, padding='same', return_sequences=True,
                            activation=lrelu, dropout=0.2, name='convlstm2d_1')(inputs)
        # layer1 = BatchNormalization()(layer1)

        if self.is_seasonal_circle:
            sc_input = Input(shape=(1,), dtype='int32')
            sc_layer = Embedding(12, 3 * 20 * 27 * 16, input_length=1)(sc_input)
            sc_layer = Reshape((3, 20, 27, 16))(sc_layer)
            layer1 = concatenate([layer1, sc_layer], axis=4)

        layer2 = ConvLSTM2D(filters=32, kernel_size=3, strides=1, padding='same', return_sequences=True,
                            activation=lrelu, dropout=0.2, name='convlstm2d_2')(layer1)
        # print(layer1.get_shape().as_list())
        # layer2 = BatchNormalization()(layer2)
        layer3 = ConvLSTM2D(filters=32, kernel_size=3, strides=1, padding='same', return_sequences=True,
                            activation=lrelu, dropout=0.2, name='convlstm2d_3')(layer2)
        # layer3 = BatchNormalization()(layer3)
        predictions = ConvLSTM2D(filters=2, padding='same', kernel_size=3, strides=1, activation='linear'
                                 , name='convlstm2d_grid_output')(layer3)
        if self.is_nino_output:
            nino_layer = NinoLayer.get_nino_layer(name='nino_output')
            predictions_ninos = nino_layer(predictions)

        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)

        if not self.is_seasonal_circle and not self.is_nino_output:
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=adam, loss=custom_function.mean_squared_error,
                          metrics=[custom_function.root_mean_squared_error])
        elif self.is_seasonal_circle and not self.is_nino_output:
            model = Model(inputs=[inputs, sc_input], outputs=predictions)
            model.compile(optimizer=adam, loss=custom_function.mean_squared_error,
                          metrics=[custom_function.root_mean_squared_error])
        elif not self.is_seasonal_circle and self.is_nino_output:
            model = Model(inputs=inputs, outputs=[predictions, predictions_ninos])
            model.compile(optimizer=adam, loss=custom_function.mean_squared_error, loss_weights=[0.25, 0.75],
                          metrics=[custom_function.root_mean_squared_error])
        else:
            model = Model(inputs=[inputs, sc_input], outputs=[predictions, predictions_ninos])
            model.compile(optimizer=adam, loss=custom_function.mean_squared_error, loss_weights=[0.25, 0.75],
                          metrics=[custom_function.root_mean_squared_error])
        # model.summary()
        return model
