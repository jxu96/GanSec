import torch
import torch.nn as nn
from .backbone import NNModule

class Generator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 latent_size=128,
                 latent_noise_size=74):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size
        self.latent_size = latent_size
        self.latent_noise_size = latent_noise_size

        self.input_size = self.n_rows * (self.latent_noise_size + self.label_embedding_size)
        self.output_size = self.n_rows * self.n_features

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.LeakyReLU(.2, inplace=True),
            nn.BatchNorm1d(self.latent_size, 0.01),
            nn.Linear(self.latent_size, self.output_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((noise, labels), dim=2)
        x = torch.flatten(x, 1)
        x = self.model(x)
        return x.view(-1, self.n_rows, self.n_features)

    def generate_random(self, amount, device, labels):
        z = torch.randn(
            amount, self.n_rows, self.latent_noise_size, device=device)
        return self.forward(z, labels)

class Discriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 label_embedding_size=1,
                 latent_size=128):
        super().__init__(data_shape, label_shape)

        self.label_embedding_size = label_embedding_size
        self.latent_size = latent_size

        self.input_size = self.n_rows * (self.n_features + self.label_embedding_size)

        self.embedding = nn.Embedding(self.n_rows, self.label_embedding_size)
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.LeakyReLU(.2, inplace=True),
            nn.Dropout(.4),
            nn.Linear(self.latent_size, 1),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        # labels = labels.reshape(-1, self.n_rows, self.label_embedding_size)
        labels = self.embedding(labels.to(torch.int))
        x = torch.cat((data, labels), dim=2)
        x = torch.flatten(x, 1)
        return self.model(x)

class LabeledDiscriminator(NNModule):
    def __init__(self,
                 data_shape,
                 label_shape,
                 latent_size=128):
        super().__init__(data_shape, label_shape)

        self.input_size = self.n_rows * self.n_features
        self.latent_size = latent_size

        self.latent = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.LeakyReLU(.2, inplace=True),
            nn.Dropout(.4),
        )
        self.fc_dis = nn.Linear(self.latent_size, 1)
        self.fc_clf = nn.Linear(self.latent_size, self.n_rows*self.n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = torch.flatten(data, 1)
        x = self.latent(x)
        pred_dis = self.sigmoid(self.fc_dis(x))
        pred_clf = self.sigmoid(self.fc_clf(x))
        return pred_dis, pred_clf

class Classifier(NNModule):
    def __init__(self, 
                 data_shape, 
                 label_shape,
                 latent_size=128):
        super().__init__(data_shape, label_shape)

        self.input_size = self.n_rows * self.n_features
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size, .01),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(self.latent_size, self.n_rows*self.n_labels),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = torch.flatten(data, 1)
        return self.model(x)
