"""
For the first test to train neural networks for zc model using dense layers.
This version the training data and testing data are together.
Input is random ssta and h1a, //for 12 different initial month,integrate for 1 month.
for same initial month,integrate for 12 month, and using every month.
"""
# Third-party libraries
from keras.layers import Input, Dense, Dropout, LeakyReLU, Concatenate
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import numpy as np

# My libraries
from network_zc.tools import file_helper

# some initial parameter
training_num = 13000
training_num2 = 12000
testing_num = 0
model_name = 'dense_model_sstaha_3'
epochs = 100
batch_size = 128


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def get_training_index(training_num1):
    test = [x1 for x1 in range(0, training_num1)]
    i = 0
    length = training_num1
    while i < length:
        for j in range(12):
            i = i + 1
        del test[i]
        length = length - 1
    return test


def get_testing_index(training_num1):
    test = [x1 for x1 in range(0, training_num1)]
    i = 0
    del test[0]
    length = training_num1 - 1
    while i < length:
        for j in range(12):
            i = i + 1
        if i < length:
            del test[i]
            length = length - 1
    return test


if __name__ == '__main__':
    # To define the model
    # l2_lamda = 0.00000
    inputs = Input(shape=(1080,), name='sstah1a_input')
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lamda),
    #           bias_regularizer=regularizers.l2(l2_lamda))(inputs)
    # x = Dropout(0.2)(x)
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lamda),
    #           bias_regularizer=regularizers.l2(l2_lamda))(x)
    # x = Dropout(0.2)(x)

    # for second input
    month_input = Input(shape=(1,), name='month_input')
    x1 = Dense(12, kernel_initializer='glorot_normal')(month_input)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.3)(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(512, kernel_initializer='glorot_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.3)(x1)
    x1 = Dropout(0.2)(x1)
    concatenated = Concatenate()([inputs, month_input])

    x = BatchNormalization()(concatenated)
    x = Dense(2190, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Dense(1080, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Dense(512, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    predictions = Dense(540, activation='linear')(x)

    model = Model(inputs=[inputs, month_input], outputs=predictions)
    # model = Model(inputs=inputs, outputs=predictions)

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=mean_squared_error, metrics=[mean_squared_error, root_mean_squared_error,
                                                                    mean_absolute_error])
    # to train model
    training_data, testing_data = file_helper.load_sstha_for_conv2d(training_num-1)
    data_x = np.reshape(training_data[get_training_index(training_num)], (training_num2, 1080))
    data_y = np.reshape(training_data[get_testing_index(training_num), :, :, 0], (training_num2, 540))

    month_data = [x + 1 for x in range(12)]
    month_data = month_data * 1000
    month_data = np.array(month_data)

    # shuffle the data for valid
    # permutation = np.random.permutation(data_x.shape[0])
    # shuffled_x = data_x[permutation]
    # shuffled_y = data_y[permutation]
    tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    # train_hist = model.fit(data_x, data_y, batch_size=batch_size, epochs=epochs, verbose=2,
    #                        callbacks=[tesorboard], validation_split=0.1)
    train_hist = model.fit([data_x, month_data], data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[tesorboard], validation_split=0.1)

    # To save the model and logs
    model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper.find_logs(model_name+'_train'), 'w') as f:
        f.write(str(train_hist.history))

    # To predict the result
    # predict_data = np.empty([1, 540])
    # predict_data[0] = data_loader.read_data(3170)
    # data_loader.write_data(3170, model.predict(predict_data)[0])

