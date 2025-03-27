import torch
import torch.nn as nn
from backbone import GANModule


class DNNGenerator(GANModule):
    def __init__(self,
                 data_shape: tuple[int, int],
                 label_embedding_shape: tuple[int, int],
                 n_labels: int,
                 latent_noise_size: int
                 ):
        super().__init__(data_shape, label_embedding_shape, n_labels)

        self.latent_noise_size = latent_noise_size
        self.input_size = self.latent_noise_size + \
            self.label_embedding_shape[1]
        self.output_size = self.data_shape[0] * self.data_shape[1]
        self.model = nn.Sequential(
            # Input is noise vector + label embedding
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, 0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.output_size),
            nn.Tanh()  # Output values between -1 and 1
            # nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # Concatenate noise vector and label embeddings
        gen_input = torch.cat(
            (noise, self.label_embedding(labels.to(torch.int).flatten())), -1)
        data = self.model(gen_input)
        # data = (data - .5) * 2
        return torch.reshape(data, self.data_shape)

    def generate_random(self, amount, device, labels=None):
        z = torch.randn(amount, self.latent_noise_size, device=device)
        if labels is None:
            labels = torch.randint(0, self.n_labels, (amount,), device=device)
        return self.forward(z, labels), labels


class DNNDiscriminator(GANModule):
    def __init__(self,
                 data_shape: tuple[int, int],
                 label_embedding_shape: tuple[int, int],
                 n_labels: int
                 ):
        super().__init__(data_shape, label_embedding_shape, n_labels)

        self.input_size = self.data_shape[0] * \
            self.data_shape[1] + self.label_embedding_shape[1]
        self.model = nn.Sequential(
            # Input is image pixels + label embedding
            nn.Linear(self.input_size, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.2),
            # nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            # nn.Linear(8, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

        # self.lstm = nn.LSTM()

    def forward(self, data, labels):
        # Concatenate image and label embeddings
        d_input = torch.cat((data, self.label_embedding(
            labels.to(torch.int).flatten())), -1)
        validity = self.model(d_input)
        return validity


class Generator(nn.Module):
    def __init__(self, **config):
        super(Generator, self).__init__()

        self.input_size = config.get('input_size')
        self.latent_size = config.get('latent_size', 100)
        self.n_classes = config.get('n_classes', 2)
        self.embedding_size = config.get('embedding_size', 1)

        # Embedding layer for labels
        self.label_embedding = nn.Embedding(
            self.n_classes, self.embedding_size)

        self.model = nn.Sequential(
            # Input is noise vector + label embedding
            nn.Linear(self.latent_size + self.embedding_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, 0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.input_size),
            nn.Tanh()  # Output values between -1 and 1
            # nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # Concatenate noise vector and label embeddings
        gen_input = torch.cat(
            (noise, self.label_embedding(labels.to(torch.int).flatten())), -1)
        data = self.model(gen_input)
        # data = (data - .5) * 2
        return data

    def generate_random(self, amount, device, labels=None):
        z = torch.randn(amount, self.latent_size, device=device)
        if labels is None:
            labels = torch.randint(0, self.n_classes, (amount,), device=device)
        return self.forward(z, labels), labels


class Discriminator(nn.Module):
    def __init__(self, **config):
        super(Discriminator, self).__init__()

        self.input_size = config.get('input_size')
        self.n_classes = config.get('n_classes', 2)
        self.embedding_size = config.get('embedding_size', 1)

        self.label_embedding = nn.Embedding(
            self.n_classes, self.embedding_size)  # Embedding layer for labels

        self.model = nn.Sequential(
            # Input is image pixels + label embedding
            nn.Linear(self.input_size + self.embedding_size, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.2),
            # nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            # nn.Linear(8, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, data, labels):
        # Concatenate image and label embeddings
        d_input = torch.cat((data, self.label_embedding(
            labels.to(torch.int).flatten())), -1)
        validity = self.model(d_input)
        return validity
