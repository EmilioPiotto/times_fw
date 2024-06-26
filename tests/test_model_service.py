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

def test_hyperparameter_optimization(tensor_sample_data):
    x_train, y_train, x_test, y_test = tensor_sample_data
    model_params_space = {
        'input_size': 10,
        'output_size': 1
    }
    best_hyperparams = ModelService.hyperparameter_optimization(LinRegNN, x_train, y_train, x_test, y_test, model_params_space, max_evals=2)

    assert best_hyperparams['learning_rate'] > 0
    assert best_hyperparams['epochs'] > 0

def test_train_loop(lin_reg_nn, tensor_sample_data):
    x_train, y_train, _, _ = tensor_sample_data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lin_reg_nn.parameters(), lr=0.001)
    model_service = ModelService()
    loss = model_service.train_loop(lin_reg_nn, criterion, optimizer, x_train, y_train, epochs=2)
    assert loss > 0

def test_evaluation(lin_reg_nn, tensor_sample_data):
    _, _, x_test, y_test = tensor_sample_data
    criterion = nn.MSELoss()
    model_service = ModelService()
    loss = model_service.evaluation(lin_reg_nn, x_test, y_test, criterion)
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

def test_predict_with_confidence_interval(lin_reg_nn, tensor_sample_data):
    x_train, y_train, x_test, y_test = tensor_sample_data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lin_reg_nn.parameters(), lr=0.001)
    model_service = ModelService()

    # Train the model
    model_service.train_loop(lin_reg_nn, criterion, optimizer, x_train, y_train, epochs=2)

    # Test prediction with confidence interval
    predicted_mean, lower_bound, upper_bound = model_service.predict_with_confidence_interval(lin_reg_nn, x_test[:5])

    assert predicted_mean.shape == torch.Size([5, 1, 1])
    assert lower_bound.shape == torch.Size([5, 1, 1])
    assert upper_bound.shape == torch.Size([5, 1, 1])

    # Ensure that the lower bound is less than the predicted mean and the predicted mean is less than the upper bound
    assert torch.all(lower_bound <= predicted_mean)
    assert torch.all(predicted_mean <= upper_bound)