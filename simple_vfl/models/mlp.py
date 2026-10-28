import torch.nn as nn


class MLP4_Credit(nn.Module):
    """Small four-layer encoder for the credit-card experiment.

    The credit table has 11 or 12 inputs, so the local encoder is deliberately
    narrow. Dimensions are halved at every layer and the output width is set by
    the caller (``args.hidden_dim``).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        widths = (input_dim, 64, 32, 16, output_dim)
        layers = []
        for left, right in zip(widths[:-1], widths[1:], strict=True):
            layers.append(nn.Linear(left, right))
            if right != output_dim:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
