import torch
import torch.nn as nn


class ModelService(nn.Module):
    def __init__(self):
        super(ModelService, self).__init__()

    def forward(self, x):
        pass
    
    def train_loop(self, model, criterion, optimizer, x_train, y_train, epochs):
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for i in range(len(x_train)):
                optimizer.zero_grad()
                output = model(x_train[i])
                loss = criterion(output, y_train[i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
            average_loss = total_loss / len(x_train)
        
            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')
    
    def evaluation(self, model, x_test, y_test, criterion):
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for i in range(len(x_test)):
                output = model(x_test[i])
                loss = criterion(output, y_test[i])
                total_loss += loss.item()
        
            average_loss = total_loss / len(x_test)
            print(f'Test Loss: {average_loss:.4f}')
    


class SimpleNN(ModelService):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelService, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @staticmethod
    def reshape_input(s):
        return s.permute(0, 2, 1)
    
class SimpleLSTM(ModelService):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelService, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the output from the last time step
        return output
    
    @staticmethod
    def reshape_input(s):
        return s.unsqueeze(1)