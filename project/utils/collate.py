import torch
from typing import List, Tuple
from torch import Tensor


def collate_fn_next_token(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    # batch: list of tuples (word_tensor, tag_tensor)
    # Get input sentences
    moves = [item[0] for item in batch]
    # Get labels
    next_move = [item[1] for item in batch]
    # Get maximum length in the batch
    lengths = [len(s) for s in moves]
    max_len = max(lengths)

    # Pad shorter sentences to let the input tensors all have the same size
    padded_moves = []
    padded_next_move = []
    for m, n_m in zip(moves, next_move):
        pad_len = max_len - len(m)
        # Padding uses index 0 both for words and labels
        # NOTE: HARD CODED TO 1960 REPLACE WITH VOCAB SIZE
        padded_moves.append(
            torch.cat([m, 1960 * torch.ones(pad_len, dtype=torch.int)]))
        padded_next_move.append(
            torch.cat([n_m, 1960 * torch.ones(pad_len, dtype=torch.int)]))

    return torch.stack(padded_moves), torch.stack(padded_next_move)
