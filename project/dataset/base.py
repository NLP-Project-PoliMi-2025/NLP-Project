import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset

from project.db_utils import connect_chess_db, fetch_games


class _ChessDataset(Dataset):
    def __init__(self, database: str, encoder: Word2Vec):
        self.database = database
        self.name = type(self).__name__
        self.encoder = encoder
        self.conn, self.cursor = connect_chess_db(self.database)

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

    def encode_moves(self, moves: list[str]) -> np.ndarray:
        """
        Args:
            moves (list[str]): list of moves
        """
        features = self.encoder.wv[moves]
        features = torch.from_numpy(features)
        return features
