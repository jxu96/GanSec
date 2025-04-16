import torch
import torch.nn as nn
from .backbone import NNModule

class Generator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 latent_noise_size=74,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size
        self.latent_noise_size = latent_noise_size
        self.lstm_hidden_size = lstm_hidden_size

        self.input_dim = self.latent_noise_size + self.label_embedding_size

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, self.n_features)

    def forward(self, noise, labels):
        # labels = labels.reshape(-1,
        #                         self.n_rows,
        #                         self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((noise, labels), dim=2)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

    def generate_random(self, amount, device, labels):
        z = torch.randn(
            amount, self.n_rows, self.latent_noise_size, device=device)
        return self.forward(z, labels)

class Discriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_shape)
        
        self.label_embedding_size = label_embedding_size
        self.lstm_hidden_size = lstm_hidden_size

        self.input_dim = self.n_features + self.label_embedding_size

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, labels):
        # labels = labels.reshape(-1,
        #                         self.n_rows,
        #                         self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((data, labels), dim=2)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        out = self.fc(h_last)
        out = self.sigmoid(out)
        return out

class LabeledDiscriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_shape)

        self.lstm_hidden_size = lstm_hidden_size
        
        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 1)
        self.fc2 = nn.Linear(self.lstm_hidden_size, self.n_rows)
        self.sigmoid = nn.Sigmoid()

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        # data = data.reshape(-1, self.n_rows, self.n_features)
        _, (h_n, _) = self.lstm(data)
        h_last = h_n[-1]
        pred_real = self.fc1(h_last)
        pred_real = self.sigmoid(pred_real)
        pred_label = self.fc2(h_last)
        pred_label = self.sigmoid(pred_label)

        return pred_real, pred_label

class Classifier(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_shape)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, self.n_rows * self.n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        h0 = torch.zeros(self.lstm_num_layers, data.size(0), self.lstm_hidden_size).to(data.device)
        c0 = torch.zeros(self.lstm_num_layers, data.size(0), self.lstm_hidden_size).to(data.device)
        
        out, _ = self.lstm(data, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
