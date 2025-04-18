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
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(32, .01),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(64, .01),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*self.hidden_size, self.output_size),
            nn.Tanh(),
        ).to(self.device)

    def forward(self, noise, labels):
        # labels = labels.repeat(1, self.label_embedding_size).reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        # print("generator", noise.shape, labels.shape)
        x = torch.cat((noise, labels), dim=2).unsqueeze(dim=1)
        x = self.model(x)
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
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(16, .01),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(32, .01),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*self.output_size, 128),
            nn.LeakyReLU(.2, inplace=True),
            nn.Dropout(.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, data, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((data, labels), dim=2).unsqueeze(dim=1)

        return self.model(x)

class LabeledDiscriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.output_size = self.n_rows * self.n_features

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(16, .01),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(32, .01),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*self.output_size, 128),
            nn.LeakyReLU(.2, inplace=True),
            nn.Dropout(.4),
        ).to(self.device)

        self.fc_dis = nn.Linear(128, 1)
        self.fc_clf = nn.Linear(128, self.n_rows*self.n_labels)
        self.sigmod = nn.Sigmoid()

    def forward(self, data):
        data = data.reshape(-1, 1, self.n_rows, self.n_features)
        x = self.model(data)
        pred_dis = self.sigmod(self.fc_dis(x))
        pred_clf = self.sigmod(self.fc_clf(x))

        return pred_dis, pred_clf

class Classifier(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape):
        super().__init__(data_shape, label_shape)

        self.output_size = self.n_rows * self.n_features

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(32, .01),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm2d(64, .01),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*self.output_size, self.n_rows*self.n_labels),
            nn.Sigmoid(),
        )

    def forward(self, data):
        data = data.reshape(-1, 1, self.n_rows, self.n_features)
        
        return self.model(data)
