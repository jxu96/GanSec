import torch
import torch.nn as nn
from .backbone import NNModule

class Generator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 latent_noise_size=74):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size
        self.latent_noise_size = latent_noise_size

        self.input_size = self.n_rows * (self.latent_noise_size + self.label_embedding_size)
        self.output_size = self.n_rows * self.n_features

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, noise, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((noise, labels), dim=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.n_rows, self.n_features)

    def generate_random(self, amount, device, labels):
        z = torch.randn(
            amount, self.n_rows, self.latent_noise_size, device=device)
        return self.forward(z, labels)

class Discriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size

        self.input_size = self.n_rows * (self.n_features + self.label_embedding_size)

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((data, labels), dim=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

class LabeledDiscriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.input_size = self.n_rows * self.n_features

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128, self.n_rows*self.n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = torch.flatten(data, 1)
        x = torch.relu(self.fc1(x))
        pred_real = self.sigmoid(self.fc2(x))
        pred_label = self.sigmoid(self.fc3(x))
        return pred_real, pred_label

class Classifier(NNModule):
    def __init__(self, 
                 data_shape, 
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.input_size = self.n_rows * self.n_features

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, self.n_rows*self.n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = torch.flatten(data, 1)
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))
