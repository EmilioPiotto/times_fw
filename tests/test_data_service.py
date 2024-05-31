import torch

def test_data_initialization_and_sample(data_processor, raw_data):
    assert raw_data is not None
    assert len(raw_data) == 50

def test_normalize_data(normalized_data):
    data_normalized, scaler = normalized_data
    assert data_normalized is not None
    assert scaler is not None
    assert len(data_normalized) == 50

def test_create_sequences(sequences):
    x_data, y_data = sequences
    assert x_data is not None
    assert y_data is not None
    assert len(x_data) == len(y_data)

def test_split_data(split_data):
    x_train, x_test, y_train, y_test = split_data
    assert x_train is not None
    assert x_test is not None
    assert y_train is not None
    assert y_test is not None

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_tensor_conversion_and_shape(tensor_sample_data):
    x_train, y_train, x_test, y_test = tensor_sample_data
    
    assert x_train.shape == torch.Size([32, 10, 1])
    assert y_train.shape == torch.Size([32, 1, 1])
    assert x_test.shape == torch.Size([8, 10, 1])
    assert y_test.shape == torch.Size([8, 1, 1])
