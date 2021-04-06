from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError

import numpy as np


class LSTMModel():
    MAX_EPOCHS = 10

    def __init__(self):
        self.regressor = Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            # Shape => [batch, time, features]
            Dense(1)
        ])

    def create_model_entries(data_scaled_features, data_scaled_target):
        X_data = []
        y_data = []
        num_entries = data_scaled_features.shape[0]
        for i in range(0, num_entries):
            X_data.append(data_scaled_features[i])
            y_data.append(data_scaled_target[i:i+1])
        X_data, y_data = np.array(X_data), np.array(y_data)
        
        X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
        
        return X_data, y_data


    def compile_and_fit(self, window, patience=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        mode='min')

        self.regressor.compile(loss=MeanSquaredError(),
                        optimizer=Adam(),
                        metrics=[MeanAbsoluteError()])

        history = self.regressor.fit(window.train, epochs=self.MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history
