import matplotlib.pyplot as plt

class VisualizationService:
    def plot_predictions(self, data, predictions, seq_length):
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Original Data')
        plt.plot(range(seq_length, len(predictions) + seq_length), predictions, label='Predicted Data')
        plt.legend()
        plt.show()
