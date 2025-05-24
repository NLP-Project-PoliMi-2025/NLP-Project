from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from nlpChess.dataset.base import _ChessDataset
import torch
import sqlite3
from nlpChess.db_utils import fetch_games, fetch_games_with_moves, fetch_moves
from nlpChess.utils.info import process_runindicator


class BoardStatePredictionDataset(_ChessDataset):
    def __init__(self, database, encoder, vocab_table):
        super().__init__(database, encoder, vocab_table)

        self.fen_vocab, self.fen_vocab_map = self.get_fen_map()

        # get game lengths
        query = """
        SELECT DISTINCT board_fen_id, move_number, game_id
        FROM moves
        """
        df = pd.read_sql_query(query, con=self.conn)
        self.game_lengths = df.groupby("game_id").aggregate("max")[
            "move_number"].values
        self.game_bins = self.game_lengths.cumsum()

    def get_fen_map(self):
        pieces = "prnbqk"
        extra_symbols = " W-09/"
        sos = "SOS"  # start of sequence
        eos = "EOS"  # end of sequence
        board_vocab = "12345678" + pieces.lower() + pieces.upper() + extra_symbols
        fen_vocab = np.array([*list(board_vocab), sos, eos])
        fen_symbol_map = dict(zip(fen_vocab, np.arange(len(fen_vocab))))
        return fen_vocab, fen_symbol_map

    def encode_board(self, board_fen: str) -> torch.Tensor:
        mask = (
            self.fen_vocab[None]
            == np.array(["SOS"] + list(board_fen) + ["EOS"])[..., None]
        )
        _, idx = np.where(mask)
        return torch.from_numpy(idx)

    def __len__(self):
        return self.game_bins[-1]

    def __getitem__(self, index):
        game_id = np.digitize(index, self.game_bins)
        if game_id > 0:
            move_idx = index - self.game_bins[game_id - 1]
        else:
            move_idx = index

        query = f"""
                    SELECT mc.move, m.board_fen_id
                    FROM moves m
                    Join move_collection mc on mc.id = m.move_id
                    Where m.game_id = {game_id + 1} AND m.move_number <= {move_idx + 1} 
                    ORDER BY m.move_number
                """
        move_df = pd.read_sql_query(query, con=self.conn)
        moves = self.encode_moves(move_df["move"].values)

        query = f"""
                    SELECT board_fen
                    FROM board_states
                    Where id = {move_df.iloc[-1]["board_fen_id"]} 
                """
        fen = pd.read_sql_query(query, con=self.conn)["board_fen"].values[0]
        fen = self.encode_board(fen)
        return moves, fen


class NextTokenDataset(_ChessDataset):
    def __init__(
        self, database, encoder, vocab_table, game_ids: List[int], use_ram: bool = False
    ):
        super().__init__(database, encoder, vocab_table)
        self.game_ids = game_ids
        self.use_ram = use_ram

        if self.use_ram:
            query = f"""
                SELECT mc.move, m.move_number, m.game_id
                FROM moves m
                JOIN move_collection mc ON m.move_id = mc.id
                WHERE m.game_id IN {tuple(game_ids)}
                ORDER BY m.game_id, m.move_number
            """
            with process_runindicator(f"{self.__repr__()}: Loading moves into RAM"):
                self.df = pd.read_sql_query(
                    query, con=sqlite3.connect(self.database))
            self.max_move_number = self.df["move_number"].max()

    def _load_from_db(self, index: int) -> np.ndarray:
        game_id = self.game_ids[index]
        query = f"""
                    SELECT mc.move
                    FROM moves m
                    Join move_collection mc on mc.id = m.move_id
                    Where m.game_id = {game_id} 
                    ORDER BY m.move_number
                """
        move_df = pd.read_sql_query(query, con=self.conn)
        moves = move_df["move"].values
        return moves

    def _load_from_ram(self, index: int) -> np.ndarray:
        game_id = self.game_ids[index]
        moves = (
            self.df[self.df["game_id"] == game_id]
            .sort_values("move_number")["move"]
            .values
        )
        return moves

    def __len__(self):
        return len(self.game_ids)

    def __getitem__(self, index):
        # index = game_id
        if self.use_ram:
            moves = self._load_from_ram(index)
        else:
            moves = self._load_from_db(index)
        moves = self.encode_moves(moves)
        x = moves[:-1]
        y = moves[1:]
        return x, y


if __name__ == "__main__":
    from gensim.models import Word2Vec

    example_fen = "8/2k5/1p6/p2QP3/8/2r5/P7/1K6 w - - 1 43"
    w2v = Word2Vec.load("models/word2vec.model")
    ds = BoardStatePredictionDataset("data/chess_games_1.db", w2v)

    w2v = Word2Vec.load("models/word2vec.model")
    ds = NextTokenDataset("data/chess_games_1.db", w2v)
