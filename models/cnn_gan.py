import torch
import torch.nn as nn
from .backbone import GANModule


class CNNGenerator(GANModule):
    def __init__(self,
                 data_shape=[5, 74],
                 label_embedding_shape=[5, 1],
                 n_labels=5,
                 latent_noise_size=74
                 ):
        super().__init__(data_shape, label_embedding_shape, n_labels)

        self.latent_noise_size = latent_noise_size
        self.hidden_size = self.data_shape[0] * (self.data_shape[1]+1)
        self.output_size = self.data_shape[0] * self.data_shape[1]

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*self.hidden_size, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, noise, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0], self.label_embedding_shape[1])
        # print("generator", noise.shape, labels.shape)
        x = torch.cat((noise, labels), dim=2).unsqueeze(dim=1)
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.reshape(x, [-1, self.data_shape[0], self.data_shape[1]])

    def generate_random(self, amount, device, labels=None):
        z = torch.randn(
            amount, self.data_shape[0], self.latent_noise_size, device=device)
        return self.forward(z, labels)


class CNNDiscriminator(GANModule):
    def __init__(self,
                 data_shape=[5, 75],
                 label_embedding_shape=[5, 1],
                 n_labels=5
                 ):
        super().__init__(data_shape, label_embedding_shape, n_labels)

        self.output_size = self.data_shape[0] * self.data_shape[1]

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*self.output_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, data, labels):
        labels = labels.reshape(-1,
                                self.label_embedding_shape[0], self.label_embedding_shape[1])
        # print("discriminator", data.shape, labels.shape)
        x = torch.cat((data, labels), dim=2).unsqueeze(dim=1)
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmod(x)

        return x
