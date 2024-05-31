import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class DataProcessingService:
    def __init__(self, seq_length:int = 10,pred_window:int = 1, batch_size:int = 10):
        """
        pred_window: (prediction window) how many periods ahead will be forcasted
        """
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.pred_window = pred_window

    def normalize_data(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(data.reshape(-1, 1))
        return data_normalized, scaler

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            s = i+self.seq_length
            x = data[i:s]
            y = data[s:s+self.pred_window]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def split_data(self, x_data, y_data, train_ratio=0.8):
        train_size = int(len(x_data) * train_ratio)
        x_train, x_test = x_data[:train_size], x_data[train_size:]
        y_train, y_test = y_data[:train_size], y_data[train_size:]
        return x_train, x_test, y_train, y_test

    @staticmethod
    def get_sample_data(length):
        # Linear data with some noise
        start = torch.rand(1).item()
        step = torch.rand(1).item()
        var = step*4

        array = torch.arange(start, start + step * length, step)
        random_values = torch.FloatTensor(length).uniform_(-var, var)

        return array + random_values
