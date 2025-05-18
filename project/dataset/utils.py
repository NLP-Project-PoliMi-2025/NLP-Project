import torch


def create_triangle(tokens: torch.Tensor) -> torch.Tensor:
    n_tokens, _ = tokens.shape
    mask = torch.tril(torch.ones((n_tokens, n_tokens)))[..., None]
    return tokens[None] * mask
