"""
For the first test to train neural networks for zc model using convolution layers.
"""
# Third-party libraries
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, LocallyConnected2D
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import numpy as np

# My libraries
from network_zc.tools import file_helper

# some initial parameter
training_num = 7000
testing_num = 0
model_name = 'convolution_model_alex'
epochs = 300
batch_size = 128


if __name__ == '__main__':
    # To define the model
    inputs = Input(shape=(20, 27, 2))
    layer1 = BatchNormalization()(inputs)
    # 20x27 to 10x14
    # for conv2d, the padding is same for layer1 and layer2,and the kernel_sizes are 7 and 5.
    layer1 = LocallyConnected2D(filters=32, kernel_size=5, strides=2, padding='valid')(layer1)
    layer1 = BatchNormalization()(layer1)
    layer1 = LeakyReLU(alpha=0.3)(layer1)
    # layer1 = Dropout(0.2)(layer1)
    # 10x14 to 10x14
    layer2 = LocallyConnected2D(filters=64, kernel_size=3, strides=1, padding='valid')(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = LeakyReLU(alpha=0.3)(layer2)
    # layer2 = Dropout(0.2)(layer2)
    # print(layer2.get_shape().as_list())
    # 10x14 to 8x12
    layer3 = LocallyConnected2D(filters=64, kernel_size=3, strides=1, padding='valid')(layer2)
    layer3 = BatchNormalization()(layer3)
    layer3 = LeakyReLU(alpha=0.3)(layer3)
    # 8x12 to 6x10
    layer4 = LocallyConnected2D(filters=80, kernel_size=3, strides=1, padding='valid')(layer3)
    layer4 = BatchNormalization()(layer4)
    layer4 = LeakyReLU(alpha=0.3)(layer4)
    layer5 = Flatten()(layer4)
    layer5 = Dense(4096)(layer5)
    layer5 = LeakyReLU(alpha=0.3)(layer5)
    layer5 = Dropout(0.3)(layer5)
    layer6 = Dense(2048)(layer5)
    layer6 = LeakyReLU(alpha=0.3)(layer6)
    layer6 = Dropout(0.3)(layer6)
    predictions = Dense(540, activation='linear')(layer6)
    # predictions = Reshape((20, 27))(layer7)
    model = Model(inputs=inputs, outputs=predictions)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mse', metrics=['mse'])
    # to train model
    # the data for training is only ssta
    # training_data, testing_data = file_helper.load_sst_for_conv2d(training_num, testing_num)
    # the data for training is ssta and ha
    training_data, testing_data = file_helper.load_sstha_for_conv2d(training_num, testing_num)
    data_x = training_data[0:-1]
    data_y = np.reshape(training_data[1:, :, :, 0], (training_num, 540))
    # shuffle the data for valid
    permutation = np.random.permutation(data_x.shape[0])
    shuffled_x = data_x[permutation]
    shuffled_y = data_y[permutation]
    tesorboard = TensorBoard('..\..\model\\tensorboard\\' + model_name)
    train_hist = model.fit(shuffled_x, shuffled_y, batch_size=batch_size, epochs=epochs, verbose=2,
                           callbacks=[tesorboard], validation_split=0.1)

    # To save the model and logs
    model.save('..\..\model\\' + model_name + '.h5')
    with open(file_helper.find_logs(model_name+'_train'), 'w') as f:
        f.write(str(train_hist.history))

    # To predict the result
    # predict_data = np.empty([1, 540])
    # predict_data[0] = data_loader.read_data(3170)
    # data_loader.write_data(3170, model.predict(predict_data)[0])

