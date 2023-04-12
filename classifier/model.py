import json
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger

class model:
    def __init__(self, data_path: str, output_path: str, num_keywords: int) -> None:
        """
        data_path: path to the data folder
        output_path: path to the output folder to save the model
        num_keywords: number of keywords to be used for training
        """
        self.data_path = data_path
        self.output_path = output_path
        self.num_keywords = num_keywords

    def load_data(self, test_size=0.1, validation_size=0.1) -> tuple:
        """
        Loads training, validation, and test data from json file.
        test_size: percentage of data to be used for testing
        validation_size: percentage of data to be used for validation
        """

        # load data
        with open(self.data_path, "r") as fp:
            data = json.load(fp)
        X, y = np.array(data["MFCCs"]), np.array(data["labels"])

        # create train, validation and test splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        # convert inputs from 2D arrays to 3D arrays (CNN input format)
        # (num_segments, num_coefficients) -> (num_segments, num_coefficients, 1)
        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def build_model(self, input_shape: tuple, learning_rate: float = 0.0001, loss: str = 'sparse_categorical_crossentropy') -> keras.Model:
        """
        Builds, compiles, and returns a CNN model.
        input_shape: shape of input data
        learning_rate: learning rate for the optimizer
        loss: loss function to be used
        """
        
        # build network topology
        model = keras.Sequential()

        # 1st conv layer
        # the kernel_regularizer parameter is used to apply L2 regularization to the weights of the layer to avoid overfitting
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
        # batch normalization layer to normalize the activations of the previous layer at each batch
        model.add(keras.layers.BatchNormalization())
        # max pooling layer to reduce the spatial dimensions of the output (downsampling)
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 2nd conv layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 3rd conv layer
        model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # flatten output and feed it into dense layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))

        # output layer (softmax)
        model.add(keras.layers.Dense(self.num_keywords, activation='softmax'))

        # compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, decay=0.001)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # print model summary
        model.summary()

        return model


    def run(self, learning_rate: float = 0.0001, batch_size: int = 32, epochs: int = 40) -> None:
        """
        Trains the CNN model.
        learning_rate: learning rate for the optimizer
        batch_size: batch size for training
        epochs: number of epochs for training
        """

        # load train/validation/test data
        X_train, X_validation, X_test, y_train, y_validation, y_test = self.load_data()
        print("X_train.shape: {}".format(X_train.shape))

        # build the CNN net
        # input_shape: (num_segments, num_coefficients, 1)
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), learning_rate=learning_rate)
        print("Model built successfully!")

        # Save training history
        csv_logger = CSVLogger('training.log', separator=',', append=False)

        # train the CNN
        train_history = self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print("Model trained successfully!")

        # evaluate the CNN on the test set
        test_error, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print("Accuracy on test set is: {}".format(test_accuracy))
        print("Error on test set is: {}".format(test_error))

        # save CNN model to file
        self.model.save(self.output_path)
        print("Model saved successfully!")

        return train_history

if __name__ == "__main__":
    cnn_model = model(data_path="classifier/data.json", output_path="cnn_model.h5", num_keywords=30)
    history = cnn_model.run()
