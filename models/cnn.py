import torch
import torch.nn as nn
from .backbone import NNModule

class Generator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 latent_noise_size=72):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size
        self.latent_noise_size = latent_noise_size

        self.hidden_size = self.n_rows * (self.n_features+self.label_embedding_size)
        self.output_size = self.n_rows * self.n_features

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*self.hidden_size, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, noise, labels):
        # labels = labels.repeat(1, self.label_embedding_size).reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        # print("generator", noise.shape, labels.shape)
        x = torch.cat((noise, labels), dim=2).unsqueeze(dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.reshape(x, [-1, self.n_rows, self.n_features])

    def generate_random(self, amount, device, labels=None):
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
        self.output_size = self.n_rows * (self.n_features + self.n_labels)

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*self.output_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, data, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((data, labels), dim=2).unsqueeze(dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmod(x)

        return x

class LabeledDiscriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.output_size = self.n_rows * self.n_features

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*self.output_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmod = nn.Sigmoid()
        self.fc3 = nn.Linear(128, self.n_rows*self.n_labels)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        data = data.reshape(-1, 1, self.n_rows, self.n_features)
        x = torch.relu(self.conv1(data))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        y1 = self.fc2(x)
        y1 = self.sigmod(y1)
        y2 = self.fc3(x)
        y2 = self.sigmod(y2)
        return y1, y2

class Classifier(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.output_size = self.n_rows * self.n_features

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32*self.output_size, self.n_rows*self.n_labels)
        self.sigmod = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        data = data.reshape(-1, 1, self.n_rows, self.n_features)
        x = torch.relu(self.conv1(data))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.sigmod(self.fc(x))
        
        return x
