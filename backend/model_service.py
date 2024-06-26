import torch
import torch.nn as nn
import torch.optim as optim
import os
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import wandb
from dotenv import load_dotenv


class ModelService(nn.Module):
    def __init__(self):
        super(ModelService, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    def train_loop(self, model, criterion, optimizer, x_train, y_train, epochs, directory="backend/checkpoints/test_", checkpoint_path_to_resume="", wandb_project = None):
        start_epoch = 0
        if checkpoint_path_to_resume and os.path.exists(checkpoint_path_to_resume):
            start_epoch, best_loss = self._resume_training(checkpoint_path_to_resume, model, optimizer)

        if wandb_project:
            # self._wandb_login()
            wandb.init(
                        project=wandb_project,
                        config={
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "architecture": str(model.__class__).split('.')[-1].strip("'>"),
                        "dataset": "CIFAR-100",
                        "epochs": epochs,
                        "optimizer": str(optimizer.__class__).split('.')[-1].strip("'>"),
                        "criterion": str(criterion.__class__).split('.')[-1].strip("'>")
                        }
                    )

        x_train = model.reshape_input(x_train)
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = self._train_epoch(model, criterion, optimizer, x_train, y_train)

            average_loss = total_loss / len(x_train)

            if wandb_project:
                wandb.log({"average_loss": average_loss})
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(directory, epoch, model, optimizer, average_loss)

        if wandb_project:
            wandb.finish()
        return average_loss

    def evaluation(self, model, x_test, y_test, criterion):
        model.eval()
        total_loss = 0.0
        x_test = model.reshape_input(x_test)

        with torch.no_grad():
            for i in range(len(x_test)):
                output = model(x_test[i])
                loss = criterion(output, y_test[i])
                total_loss += loss.item()

        average_loss = total_loss / len(x_test)
        print(f'Test Loss: {average_loss:.4f}')
        return average_loss

    @staticmethod
    def reshape_input(s):
        raise NotImplementedError("Subclasses should implement this!")

    @staticmethod
    def register(model, name: str, path: str = "backend/models/"):
        full_path = os.path.join(path, name)
        try:
            torch.save(model, full_path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving the entire model: {e}")

    @staticmethod
    def load_registered_model(path):
        try:
            loaded_model = torch.load(path)
            loaded_model.eval()
            print("Model loaded successfully.")
            return loaded_model
        except Exception as e:
            print(f"Error loading the entire model: {e}")

    @staticmethod
    def hyperparameter_optimization(model_class, x_train, y_train, x_val, y_val, model_params_space, max_evals=20):
        def objective(params):
            model = model_class(**params['model_params'])
            criterion = nn.MSELoss()  # Example criterion, modify as needed
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

            # Training loop
            model_service = ModelService()
            model_service.train_loop(model, criterion, optimizer, x_train, y_train, params['epochs'])

            # Evaluation
            val_loss = model_service.evaluation(model, x_val, y_val, criterion)
            return {'loss': val_loss, 'status': STATUS_OK}

        search_space = {
            'model_params': model_params_space,
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'epochs': scope.int(hp.quniform('epochs', 5, 20, 1))
        }

        trials = Trials()
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best)
        return best

    def _resume_training(self, checkpoint_path, model, optimizer):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}, Loss: {best_loss:.4f}")
        return start_epoch, best_loss

    def _train_epoch(self, model, criterion, optimizer, x_train, y_train):
        total_loss = 0.0
        for i in range(len(x_train)):
            optimizer.zero_grad()
            output = model(x_train[i])
            loss = criterion(output, y_train[i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _save_checkpoint(self, directory, epoch, model, optimizer, average_loss):
        checkpoint_path = f'{directory}{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    @staticmethod
    def wandb_login():
        if wandb.run is not None:
            print("There is an active connection to wandb")
        else:
            load_dotenv()
            wandb_api_key = os.getenv("WANDB_API_KEY")
            wandb.login(key=wandb_api_key)

    @staticmethod
    def predict_with_confidence_interval(model, x_data, n_samples=1000, alpha=0.05):
        """
        Generates predictions with confidence intervals.

        Parameters:
        - model: The trained model to use for predictions.
        - x_data: Input data for which to make predictions.
        - n_samples: Number of samples to draw for the prediction distribution.
        - alpha: Significance level for the confidence interval.

        Returns:
        - predicted_mean: Mean of the predictions.
        - lower_bound: Lower bound of the confidence interval.
        - upper_bound: Upper bound of the confidence interval.
        """
        model.eval()
        x_data = model.reshape_input(x_data)
        predictions = []
        noise_std = 0.08
        with torch.no_grad():
            for _ in range(n_samples):
                x_data = x_data + torch.randn_like(x_data) * noise_std # ADD NOISE
                preds = []
                for i in range(len(x_data)):
                    output = model(x_data[i])
                    preds.append(output)
                predictions.append(torch.stack(preds))

        predictions = torch.stack(predictions)
        predicted_mean = predictions.mean(dim=0)
        lower_bound = torch.quantile(predictions, alpha / 2, dim=0)
        upper_bound = torch.quantile(predictions, 1 - alpha / 2, dim=0)

        return predicted_mean, lower_bound, upper_bound


class SimpleNN(ModelService):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
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
        super(SimpleLSTM, self).__init__()
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
    

