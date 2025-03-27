
class LSTMDiscriminator(GANModule):
    def __init__(self,
                 data_shape: tuple[int, int],
                 label_embedding_shape: tuple[int, int],
                 n_labels: int,
                 hidden_size: int,
                 n_layers: int
                 ):
        super().__init__(data_shape, label_embedding_shape, n_labels)

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.n_layers, batch_first=True)
        self.net = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape([-1, 1, x.size(1)])
        h0 = torch.zeros(self.n_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.net(out[:, -1, :])
        return out
