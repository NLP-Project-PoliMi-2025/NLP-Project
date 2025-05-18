from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from einops import rearrange


class LogHiddenActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, torch.log(x + 0.5), -self.softplus.forward(-x))


class MinimalRNN(Module):
    def _preprocessing(self, x: Tensor, h_0: Tensor) -> Tuple[Tensor, Tensor]:
        if h_0 is None:
            batch_size = x.shape[0]
            # lets say we have a input of torch.ones in exponential space
            # -> torch.zeros in log space
            h_0 = torch.zeros(
                (batch_size, 1, self.hidden_dim), device=x.device)

        if self.flatten_input and len(x.shape) == 4:
            # expect multichannel input like an image: (batch_size, channels,
            # window_length, features)
            x = rearrange(x, "B C W F -> B W (F C)")
        elif self.flatten_input and len(x.shape) > 4:
            raise ValueError(
                f"No flattening logic implemented for {type(self).__name__}"
            )
        return x, h_0

    @staticmethod
    def parallel_scan_log(log_coeffs: Tensor, log_values: Tensor) -> Tensor:
        # input:
        #   log_coeffs: (batch_size, seq_len, input_size)
        #   log_values: (batch_size, seq_len + 1, input_size)
        # output:
        #   hidden_states in exp space (batch_size, seq_len, input_size)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h[:, 1:])


class MinimalGRU(MinimalRNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        encoder: Module = None,
        flatten_input: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.flatten_input = flatten_input or encoder is None

        linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        if encoder is not None:
            net = nn.Sequential(encoder, linear)
        else:
            net = linear
        self.net = net

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.hidden_log_activation = LogHiddenActivation()
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, h_0: Tensor = None) -> Tuple[Tensor, Tensor]:
        """forward in logspace for numerical stability.
        For more details please refer to section B2.1 in: https://arxiv.org/pdf/2410.01201

            Args:
                x (Tensor): (batch_size, sequence_length, n_features)
                h_0 (Tensor): (batch_size, hidden_dim). Defaults to None

            Returns: Tuple[out, hidden]
                out (Tensor): (batchpython -m pyargwriter generate-argparser --input file1.py file2.py [--output OUTPUT_DIR] [--pretty] [--log-level LOG_LEVEL]_size, output_size)
                hidden (Tensor): (batch_size, seq_len, hidden_dim)
        """
        x, h_0 = self._preprocessing(x, h_0)
        latent = self.net.forward(x)
        z = latent[..., : self.hidden_dim]
        h_tilde = latent[..., self.hidden_dim:]
        z = -self.softplus(-z)  # sigmoid(z) in log space
        z_inv = -self.softplus(z)  # (1 - z) in log space

        h_tilde = self.hidden_log_activation(h_tilde)
        h_tilde = torch.cat([h_0, z + h_tilde], dim=1)
        # if torch.any(torch.isnan(h_tilde)):
        #     print("x", x)
        #     print("h_tilde", h_tilde)
        #     print("z_inv", z_inv)
        #     # save the tensors to a text file for debugging
        #     with open("debug.txt", "w") as f:
        #         f.write(f"x: {x.cpu().detach().tolist()}\n")
        #         f.write(f"h_tilde: {h_tilde.cpu().detach().tolist()}\n")
        #         f.write(f"z_inv: {z_inv.cpu().detach().tolist()}\n")
        #         f.write(f"z: {z.cpu().detach().tolist()}\n")
        #         f.write(f"latent: {latent.cpu().detach().tolist()}\n")
        #     # raise an error
        #     raise ValueError("h_tilde contains NaN values")
        h = self.parallel_scan_log(z_inv, h_tilde)

        # transform
        out = self.out_layer.forward(h)
        return out, h
