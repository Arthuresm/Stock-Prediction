from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np


class LSTMModel():
    def __init__(self, shape):
        regressor = Sequential()

        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (shape[1], 1)))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))

        regressor.add(Dense(units = 1))

        regressor.compile(optimizer='adam', loss='mean_squared_error')

        self.regressor = regressor

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
