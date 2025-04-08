import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,
                 data_shape=[5, 72],
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.input_size = data_shape[1]

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        h0 = torch.zeros(self.lstm_num_layers, data.size(0), self.lstm_hidden_size).to(data.device)
        c0 = torch.zeros(self.lstm_num_layers, data.size(0), self.lstm_hidden_size).to(data.device)
        
        out, _ = self.lstm(data, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
