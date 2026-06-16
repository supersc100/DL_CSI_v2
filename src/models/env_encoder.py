"""Environment / large-scale parameter encoder."""
import torch
import torch.nn as nn


class EnvironmentEncoder(nn.Module):
    """Encode large-scale parameter vector into a latent vector.

    Input shape:  [B, input_dim]
    Output shape: [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = None,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
