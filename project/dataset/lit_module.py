# TODO: write custom collate function (tutorial 6)


import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule

from project.utils.collate import collate_fn_next_token
from project.dataset.seq2seq import NextTokenDataset
from project.db_utils import fetch_games
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
import sqlite3
from project.dataset.ChessDataset import ChessDataset


class NextTokenDM(LightningDataModule):
    def __init__(
        self,
        database: str,
        encoder_weights: str = None,
        batch_size: int = 1,
        num_worker: int = 1,
        use_ram: bool = True,
    ):
        super().__init__()
        self.database = database
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.num_worker = num_worker

        if self.encoder_weights is not None:
            w2v = Word2Vec.load(self.encoder_weights)
            vocab_table = None
        else:
            w2v = None
            query = """
                SELECT * 
                FROM move_collection
            """
            df = pd.read_sql_query(query, sqlite3.connect(self.database))
            vocab_table = df["move"].values

        # get all game ids
        df = fetch_games(db_path=self.database)["game_id"].values
        # shuffle ids and get first 70% as train 20% val and 10% test
        np.random.shuffle(df)
        cutoffs = (len(df) * np.array([0.7, 0.9])).astype(int)
        train_idx, val_idx, test_idx = np.split(df, cutoffs)

        self.train_set = NextTokenDataset(
            self.database,
            encoder=w2v,
            vocab_table=vocab_table,
            game_ids=train_idx,
            use_ram=use_ram,
        )
        self.val_set = NextTokenDataset(
            self.database,
            encoder=w2v,
            vocab_table=vocab_table,
            game_ids=val_idx,
            use_ram=use_ram,
        )
        self.test_set = NextTokenDataset(
            self.database,
            encoder=w2v,
            vocab_table=vocab_table,
            game_ids=test_idx,
            use_ram=use_ram,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )

    def get_vocab_size(self) -> int:
        query = """
            SELECT * FROM move_collection
        """
        df = pd.read_sql_query(query, sqlite3.connect(self.database))
        return len(df)


class SeqAnnotationDM(LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int,
        input_column: str,
        label_column: str,
        num_worker: int = 1,
    ):
        super().__init__()
        self.fit_file = train_file
        self.validate_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.input_column = input_column
        self.label_column = label_column
        self.num_worker = num_worker

        self.fit_set: ChessDataset
        self.validate_set: ChessDataset
        self.test_set: ChessDataset
        
        
    def setup_all(self):
        self.setup("fit")
        self.setup("validate")
        self.setup("test")
        
    def setup(self, stage = None):
        file = getattr(self, f"{stage}_file")
        data_set = ChessDataset(file, self.input_column, self.label_column)

        ds_name = f"{stage}_set"
        setattr(self, ds_name, data_set)

    def train_dataloader(self):
        return DataLoader(
            self.fit_set,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            collate_fn=collate_fn_next_token,
        )
