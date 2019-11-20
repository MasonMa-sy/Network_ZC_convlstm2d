"""
For the first test to train neural networks for zc model using dense layers.
This version the training data and testing data are together(not separated).
The whole data is run ZC model freely for 1005 years.
"""
# Third-party libraries
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import numpy as np

# My libraries
from network_zc.tools import file_helper_unformatted, data_preprocess, name_list

# some model name set
model_name = name_list.model_name
data_preprocess_method = name_list.data_preprocess_method

# some initial parameter
# training_start = 60
# training_num = 12060
# testing_num = 0
training_start = 0
training_num = 372
all_num = 464
testing_num = 0
epochs = 100
batch_size = 32


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    # To define the model
    # l2_lamda = 0.00000
    inputs = Input(shape=(1080,))
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lamda),
    #           bias_regularizer=regularizers.l2(l2_lamda))(inputs)
    # x = Dropout(0.2)(x)
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lamda),
    #           bias_regularizer=regularizers.l2(l2_lamda))(x)
    # x = Dropout(0.2)(x)
    x = BatchNormalization()(inputs)
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
    predictions = Dense(1080, activation='linear')(x)

    model = Model(inputs=inputs, outputs=predictions)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=mean_squared_error, metrics=[mean_squared_error, root_mean_squared_error,
                                                                    mean_absolute_error])
    # to train model
    all_data, testing_data = file_helper_unformatted.load_sstha_for_conv2d(training_start, all_num)

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

    testing_data = all_data[372:]
    training_data = all_data[:372]
    data_x = np.reshape(training_data[:-1], (training_num-1, 1080))
    data_y = np.reshape(training_data[1:], (training_num-1, 1080))
    tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    train_hist = model.fit(data_x, data_y, batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[tesorboard], validation_split=0.1)

    # To save the model and logs
    model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper_unformatted.find_logs(model_name+'_train'), 'w') as f:
        f.write(str(train_hist.history))

    data_x = np.reshape(testing_data[:-1], (all_num-training_num, 1080))
    data_y = np.reshape(testing_data[1:], (all_num-training_num, 1080))
    test_hist = model.evaluate(data_x, data_y, batch_size=batch_size, verbose=2)
    with open(file_helper_unformatted.find_logs(model_name+'_test'), 'w') as f:
        f.write(str(test_hist))
    # To predict the result
    # predict_data = np.empty([1, 540])
    # predict_data[0] = data_loader.read_data(3170)
    # data_loader.write_data(3170, model.predict(predict_data)[0])
