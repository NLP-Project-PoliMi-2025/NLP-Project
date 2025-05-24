from typing import Dict
import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset

from nlpChess.db_utils import connect_chess_db, fetch_games


class _ChessDataset(Dataset):
    def __init__(
        self, database: str, encoder: Word2Vec = None, vocab_table: np.ndarray = None
    ):
        self.database = database
        self.name = type(self).__name__
        self.encoder = encoder
        self.conn, self.cursor = connect_chess_db(self.database)

        if self.encoder is None:
            # construct lookup table for one
            assert (
                vocab_table is not None
            ), "if encoder not give `vocab_table` has to be provided"
            self.vocab_table = vocab_table

    def __getitem__(self, index: int):
        """
        Args:
            index (int): game index
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self) -> int:
        df = fetch_games(connection=self.conn, columns=["game_id"])
        return len(df)

    def __repr__(self):
        return f"ChessDataset(name={self.name})"

    def encode_moves(self, moves: np.ndarray) -> np.ndarray:
        """
        Args:
            moves (np.ndarray): list of moves
        """
        if self.encoder is not None:
            features = self.encoder.wv[moves]
            features = torch.from_numpy(features)
            return features
        else:
            idx = np.where(moves[:, None] == self.vocab_table[None])[1]
            torch_idx = torch.from_numpy(idx)
            return torch_idx
