import torch
import torch.nn as nn
import os


class ModelService(nn.Module):
    def __init__(self):
        super(ModelService, self).__init__()

    def forward(self, x):
        pass
    
    def train_loop(self, model, criterion, optimizer, x_train, y_train, epochs, directory = "backend/checkpoints/test_",checkpoint_path_to_resume = ""):
        start_epoch = 0
        if os.path.exists(checkpoint_path_to_resume):
            # Load the checkpoint to resume training
            checkpoint = torch.load(checkpoint_path_to_resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            start_epoch = checkpoint_path_to_resume.split('_')[-1]
            print(f"Resuming training from epoch {start_epoch}, Loss: {best_loss:.4f}")
            
        for epoch in range(start_epoch, epochs):
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
        
            if (epoch+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

            if (epoch + 1) % 5 == 0:
                # directory = "backend/models/test_"
                checkpoint_path = f'{directory}{epoch+1}.pth'
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_loss
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}")
        return average_loss
    
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
        return average_loss

    @staticmethod
    def reshape_input(s):
        pass
    
    @staticmethod
    def register(model, name: str, path: str = "backend/model/"):
        try:
            path = model + name
            torch.save(model, path)
            print("Entire model saved successfully.")
        except Exception as e:
            print(f"Error saving the entire model: {e}")

    @staticmethod
    def load_registered_model(path):
        try:
            loaded_model = torch.load(path)
            loaded_model.eval()
            print("Entire model loaded successfully.")
            return loaded_model
        except Exception as e:
            print(f"Error loading the entire model: {e}")



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
    

class LinRegNN(ModelService):
    def __init__(self, input_size, output_size):
        super(LinRegNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    @staticmethod
    def reshape_input(s):
        return s.permute(0, 2, 1)
