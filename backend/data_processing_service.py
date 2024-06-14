import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from statsmodels.tsa.seasonal import STL

class DataProcessingService:
    def __init__(self, seq_length: int = 10, pred_window: int = 1, batch_size: int = 10, has_date_index: bool = True):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.pred_window = pred_window
        self.has_date_index = has_date_index

    def normalize_data(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if isinstance(data, pd.Series):
            data_normalized = scaler.fit_transform(data.values.reshape(-1, 1))
        else:
            data_normalized = scaler.fit_transform(data.reshape(-1, 1))
        return data_normalized, scaler

    def decompose_series(self, data_series, seasonal=None):
        if seasonal is None:
            stl = STL(data_series)
        else:
            stl = STL(data_series, seasonal=seasonal)
        result = stl.fit()
        return result.observed, result.trend, result.seasonal, result.resid

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length - self.pred_window + 1):
            s = i + self.seq_length
            x = data[i:s]
            y = data[s:s + self.pred_window]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def split_data(self, x_data, y_data, train_ratio=0.8):
        train_size = int(len(x_data) * train_ratio)
        x_train, x_test = x_data[:train_size], x_data[train_size:]
        y_train, y_test = y_data[:train_size], y_data[train_size:]
        return x_train, x_test, y_train, y_test

    def process_multiple_series(self, time_series_list):
        """Returns observed, trend, seasonal, resid for each time series provided as input"""
        combined_data_list = []
        for series in time_series_list:
            normalized_data, scaler = self.normalize_data(series)
            if self.has_date_index and isinstance(series, pd.Series):
                data_series = pd.Series(normalized_data.squeeze(), index=series.index)
                observed, trend, seasonal, resid = self.decompose_series(data_series)
                combined_data = np.stack((observed, trend, seasonal, resid), axis=1)
            else:
                combined_data = np.stack((normalized_data.squeeze(),) * 4, axis=1)  # Replicate normalized data for all channels
            combined_data_list.append(combined_data)
        return np.array(combined_data_list)

    @staticmethod
    def get_sample_data(length):
        start = torch.rand(1).item()
        step = torch.rand(1).item()
        var = step * 4
        array = torch.arange(start, start + step * length, step)
        random_values = torch.FloatTensor(length).uniform_(-var, var)
        return array + random_values
