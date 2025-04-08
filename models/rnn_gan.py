import torch
import torch.nn as nn
from .backbone import GANModule


class Generator(GANModule):
    def __init__(self,
                 data_shape=[5, 72],
                 label_embedding_shape=[5, 1],
                 n_labels=5,
                 latent_noise_size=74,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_embedding_shape, n_labels)
        self.latent_noise_size = latent_noise_size
        self.lstm_hidden_size = lstm_hidden_size

        self.input_dim = self.latent_noise_size + self.label_embedding_shape[1]

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, self.data_shape[1])

    def forward(self, noise, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0],
                                self.label_embedding_shape[1])
        x = torch.cat((noise, labels), dim=2)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

    def generate_random(self, amount, device, labels):
        z = torch.randn(
            amount, self.data_shape[0], self.latent_noise_size, device=device)
        return self.forward(z, labels)


class Discriminator(GANModule):
    def __init__(self,
                 data_shape=[5, 72],
                 label_embedding_shape=[5, 1],
                 n_labels=5,
                 lstm_hidden_size=128,
                 lstm_num_layers=1):
        super().__init__(data_shape, label_embedding_shape, n_labels)
        self.lstm_hidden_size = lstm_hidden_size

        self.input_dim = self.data_shape[1] + self.label_embedding_shape[1]

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0],
                                self.label_embedding_shape[1])
        x = torch.cat((data, labels), dim=2)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        out = self.fc(h_last)
        out = self.sigmoid(out)
        return out
