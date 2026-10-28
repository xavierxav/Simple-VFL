import torch.nn as nn


class MLP4_Credit(nn.Module):
    """Four-layer MLP used as a local encoder for the credit-card dataset."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, self.output_dim),
        )

    def forward(self, x):
        return self.layer(x)
