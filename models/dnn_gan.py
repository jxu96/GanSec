import torch
import torch.nn as nn
from .backbone import GANModule


class Generator(GANModule):
    def __init__(self,
                 data_shape=[5, 72],
                 label_embedding_shape=[5, 1],
                 n_labels=5,
                 latent_noise_size=74):
        super().__init__(data_shape, label_embedding_shape, n_labels)
        self.latent_noise_size = latent_noise_size
        self.input_size = self.data_shape[0] * \
            (self.latent_noise_size + self.label_embedding_shape[1])
        self.output_size = self.data_shape[0] * self.data_shape[1]

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, noise, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0],
                                self.label_embedding_shape[1])
        x = torch.cat((noise, labels), dim=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.data_shape[0], self.data_shape[1])

    def generate_random(self, amount, device, labels):
        z = torch.randn(
            amount, self.data_shape[0], self.latent_noise_size, device=device)
        return self.forward(z, labels)


class Discriminator(GANModule):
    def __init__(self,
                 data_shape=[5, 72],
                 label_embedding_shape=[5, 1],
                 n_labels=5):
        super().__init__(data_shape, label_embedding_shape, n_labels)
        self.input_size = self.data_shape[0] * \
            (self.data_shape[1] + self.label_embedding_shape[1])
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0],
                                self.label_embedding_shape[1])
        x = torch.cat((data, labels), dim=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
