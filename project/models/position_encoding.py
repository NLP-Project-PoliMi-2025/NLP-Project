import numpy as np
from torch import nn
import torch


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: int) -> torch.Tensor:
        positions = np.arange(timesteps)[:, np.newaxis]  # Shape: (timesteps, 1)
        dimensions = np.arange(self.embedding_dim)[
            np.newaxis, :
        ]  # Shape: (1, embedding_dim)

        # Compute angles using sine for even indices and cosine for odd indices
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / self.embedding_dim)
        angle_rads = positions * angle_rates

        pos_encoding = np.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return torch.from_numpy(pos_encoding).float()
