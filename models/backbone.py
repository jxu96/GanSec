import torch
import torch.nn as nn

class NNModule(nn.Module):
    def __init__(self,
                 data_shape: tuple[int, int], # [n_rows, n_features]
                 label_shape: tuple[int, int], # [n_rows, n_labels]
                 ):
        assert data_shape[0] == label_shape[0]
        super().__init__()

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.n_rows = data_shape[0]
        self.n_features = data_shape[1]
        self.n_labels = label_shape[1]
        # self.label_embedding = nn.Embedding(
        #     self.label_embedding_shape[0], self.label_embedding_shape[1])

    def forward():
        pass

    def generate():
        pass
