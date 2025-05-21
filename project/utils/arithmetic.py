import torch
import numpy as np


def get_last_nonzero_indices(a: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    # Create a mask of non-zero elements
    mask = a != 0

    if isinstance(a, np.ndarray):
        # Convert to a PyTorch tensor if input is a NumPy array
        mask = torch.tensor(mask, dtype=torch.bool)

    if len(a.shape) == 1:
        # If the input is a 1D tensor, add a dimension to make it 2D
        mask = mask.unsqueeze(0)

    # Get the last non-zero index for each row
    last_nonzero_indices = (mask.cumsum(dim=1) - 1).max(dim=1)[0]
    # Convert to NumPy array if input was a NumPy array

    if len(a.shape) == 1:
        # If the input was originally 1D, return a 1D tensor
        last_nonzero_indices = last_nonzero_indices.squeeze(0)

    if isinstance(a, np.ndarray):
        last_nonzero_indices = last_nonzero_indices.numpy()

    return last_nonzero_indices


def pad_last_dim(tensor, pad_value=0):
    # Get the shape of the tensor
    shape = list(tensor.shape)
    shape[-1] = shape[-1] + 1

    # Create a new tensor with the same shape but filled with pad_value
    padded_tensor = torch.full(
        shape, pad_value, device=tensor.device, dtype=tensor.dtype
    )

    # Copy the original tensor into the new tensor
    padded_tensor[..., 1:] = tensor

    return padded_tensor
