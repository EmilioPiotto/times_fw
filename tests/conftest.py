import pytest
import torch
from data_processing_service import DataProcessingService

@pytest.fixture
def data_processor():
    return DataProcessingService(seq_length=10, pred_window=1, batch_size=10)

@pytest.fixture
def raw_data(data_processor):
    return data_processor.get_sample_data(length=50)

@pytest.fixture
def normalized_data(data_processor, raw_data):
    return data_processor.normalize_data(raw_data)

@pytest.fixture
def sequences(data_processor, normalized_data):
    data_normalized, _ = normalized_data
    return data_processor.create_sequences(data_normalized)

@pytest.fixture
def split_data(data_processor, sequences):
    x_data, y_data = sequences
    return data_processor.split_data(x_data, y_data, train_ratio=0.8)

@pytest.fixture
def tensor_sample_data(split_data):
    x_train, x_test, y_train, y_test = split_data
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

