from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

from nlpChess.dataset import _ChessDataset
from nlpChess.db_utils import fetch_games

ChessResultLabel = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
ChessTerminationLabel = {
    "CHECKMATE": 0,
    "STALEMATE": 1,
    "INSUFFICIENT_MATERIAL": 2,
    "FIVEFOLD_REPETITION": 3,
    "SEVENTYFIVE_MOVES": 4,
}
game_label_map = {"result": ChessResultLabel,
                  "termination": ChessTerminationLabel}


class ChessClassificationDataset(_ChessDataset):
    def __init__(
        self, database: str, encoder: Word2Vec, label: Literal["result", "termination"]
    ):
        super().__init__(database, encoder)
        self.label = label
        self.label_map = game_label_map[label]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:  # noqa: F821
        """
        Args:
            index (int): game index
        """
        index = index + 1  # game id starts with 1
        game_df = fetch_games(
            connection=self.conn, columns=[
                self.label], filters=[("id", "=", index)]
        )
        label = game_df[self.label].values[0]
        label = torch.tensor([self.label_map[label]], dtype=torch.long)

        if game_df.empty:
            raise IndexError(
                f"Game with index {index} not found in the database.")

        query = f"""
            SELECT mc.move 
            FROM moves m
            Join move_collection mc on mc.id = m.move_id
            Where m.game_id = {index}
            ORDER BY m.move_number
        """
        move_df = pd.read_sql_query(query, self.conn)
        features = self.encode_moves(move_df["move"].values)
        return features, label


if __name__ == "__main__":
    # Example usage
    encoder = Word2Vec.load("models/word2vec.model")
    df = ChessClassificationDataset(
        database="data/chess_games_1.db", encoder=encoder, label="result"
    )
    df[0]
