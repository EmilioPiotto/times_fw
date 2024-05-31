import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from backend.model_service import ModelService, SimpleNN, SimpleLSTM, LinRegNN


@pytest.fixture
def simple_nn():
    return SimpleNN(input_size=10, hidden_size=10, output_size=1)

@pytest.fixture
def simple_lstm():
    return SimpleLSTM(input_size=1, hidden_size=10, output_size=1)

@pytest.fixture
def lin_reg_nn():
    return LinRegNN(input_size=10, output_size=1)

def test_simple_nn_forward(simple_nn):
    x = torch.randn(32, 10)
    output = simple_nn(x)
    assert output.shape == torch.Size([32, 1])

def test_simple_lstm_forward(simple_lstm):
    x = torch.randn(32, 10, 1)
    output = simple_lstm(x)
    assert output.shape == torch.Size([32, 1])

def test_lin_reg_nn_forward(lin_reg_nn):
    x = torch.randn(32, 10)
    output = lin_reg_nn(x)
    assert output.shape == torch.Size([32, 1])

def test_train_loop(lin_reg_nn, tensor_sample_data):
    x_train, y_train, _, _ = tensor_sample_data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lin_reg_nn.parameters(), lr=0.001)
    model_service = ModelService()
    x_train_rs= lin_reg_nn.reshape_input(x_train)
    loss = model_service.train_loop(lin_reg_nn, criterion, optimizer, x_train_rs, y_train, epochs=2)
    assert loss > 0


def test_evaluation(lin_reg_nn, tensor_sample_data):
    _, _, x_test, y_test = tensor_sample_data
    criterion = nn.MSELoss()
    model_service = ModelService()
    x_test_rs= lin_reg_nn.reshape_input(x_test)
    loss = model_service.evaluation(lin_reg_nn, x_test_rs, y_test, criterion)
    assert loss > 0

def test_simple_nn_reshape_input(simple_nn, tensor_sample_data):
    x_train, _, _, _ = tensor_sample_data
    reshaped = simple_nn.reshape_input(x_train)
    assert reshaped.shape == torch.Size([32, 1, 10])

def test_simple_lstm_reshape_input(simple_lstm, tensor_sample_data):
    x_train, _, _, _ = tensor_sample_data
    reshaped = simple_lstm.reshape_input(x_train)
    assert reshaped.shape == torch.Size([32, 1, 10, 1])

def test_lin_reg_nn_reshape_input(lin_reg_nn, tensor_sample_data):
    x_train, _, _, _ = tensor_sample_data
    reshaped = lin_reg_nn.reshape_input(x_train)
    assert reshaped.shape == torch.Size([32, 1, 10])
