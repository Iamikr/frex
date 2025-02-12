from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from tensorflow.keras.models import model_from_json


def create_cnn(input_shape, kernel_size, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    chanDim = -1

    # define the model input
    inputs = Input(shape=input_shape)
    # loop over the number of filters
    x = inputs
    for f in filters:
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (kernel_size, kernel_size), padding="same")(
            x)  # kernel kernel_size x kernel_size x f, so it produces f images
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)  # FC (Fully-Connected) Layer
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    x = Dense(2, activation="softmax")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model


def run(train_X, valid_X, test_X, train_Y, valid_Y, test_Y, epochs, batch_size, filters, kernel, lr, mse_min, acc_max):
    # create our Convolutional Neural Network and then compile the model
    model = create_cnn(train_X.shape[1:], kernel, filters)
    opt = Adam(lr=lr, decay=1e-3)
    model.compile(loss=BinaryCrossentropy(), optimizer=opt)

    # train the model
    print("[INFO] training model...")

    # Create callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=np.max([epochs // 2, 30]), restore_best_weights=True)]

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=int(epochs // 5), verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0.0000001))

    model.fit(x=train_X, y=train_Y,
              validation_data=(valid_X, valid_Y),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

    print("[INFO] testing model...")
    preds = model.predict(test_X)

    acc = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y)]) / len(test_Y)
    acc0 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 0]) / len(
        test_Y[test_Y == 0])
    acc1 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 1]) / len(
        test_Y[test_Y == 1])

    if acc * acc1 > acc_max:
        model_json = model.to_json()
        with open("CNNPivot.json", "w+") as json_file:
            json_file.write(model_json)

        model.save_weights("resNet.h5")
        with open("CNNPivot.txt", "w+") as bestFile:
            bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}".format(epochs, batch_size, lr))

    return preds, acc, acc0, acc1


def runLoadedModel(test_X, test_Y):
    # load json and create model
    json_file = open('CNNPivot.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('resNet.h5')

    preds = loaded_model.predict(test_X)
    acc = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y)]) / len(test_Y)
    acc0 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 0]) / len(
        test_Y[test_Y == 0])
    acc1 = np.sum([np.argmax(pred) == true for pred, true in zip(preds, test_Y) if true == 1]) / len(
        test_Y[test_Y == 1])

    print("Accuracy: {}".format(acc))
    print("Accuracy on Pivot Points: {}".format(acc1))
    print("Accuracy on non-Pivot Points: {}".format(acc0))

    return preds
