import torch
import torch.nn as nn


class GANModule(nn.Module):
    def __init__(self,
                 data_shape: tuple[int, int],
                 label_embedding_shape: tuple[int, int],
                 n_labels: int
                 ):
        super().__init__()

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.data_shape = data_shape  # [n_rows, n_features]
        # [n_labels, n_neurons]
        self.label_embedding_shape = label_embedding_shape
        self.n_labels = n_labels
        self.label_embedding = nn.Embedding(
            self.label_embedding_shape[0], self.label_embedding_shape[1])

        # self.latent_size = config.get('latent_size', 100) # neurons for noise

    def forward():
        pass

    def generate():
        pass
