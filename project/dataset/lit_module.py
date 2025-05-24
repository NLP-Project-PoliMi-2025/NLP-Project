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

from datasets import Dataset as HuggingfaceDataset


class SeqAnnotationDM(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        input_column: str,
        label_column: str,
        datasets: HuggingfaceDataset = None,
        train_file: str = "",
        val_file: str = "",
        test_file: str = "",
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

        self.datasets: HuggingfaceDataset = datasets

        self.fit_set: ChessDataset = None
        self.validate_set: ChessDataset = None
        self.test_set: ChessDataset = None

    def get_vocab_size(self) -> tuple[int]:
        vocabSizes = []
        for column in self.test_set.inputColumns:
            vocabSizes.append(len(self.test_set.lookup_tables[column]))
        return tuple(vocabSizes)

    def get_num_labels(self) -> tuple[int]:
        numLabels = []
        for column in self.test_set.labelColumns:
            numLabels.append(len(self.test_set.lookup_tables[column]))
        return tuple(numLabels)

    def setup(self, stage=None):
        if self.fit_set is not None:
            return
        self.fit_set = ChessDataset(
            dataset=self.datasets['train'] if self.datasets else None,
            parquette_path=self.fit_file,
            inputColumns=self.input_column,
            labelColumns=self.label_column,
            lookupReference=None,
        )
        self.validate_set = ChessDataset(
            dataset=self.datasets['validation'] if self.datasets else None,
            parquette_path=self.validate_file,
            inputColumns=self.input_column,
            labelColumns=self.label_column,
            lookupReference=self.fit_set,
        )
        self.test_set = ChessDataset(
            dataset=self.datasets['test'] if self.datasets else None,
            parquette_path=self.test_file,
            inputColumns=self.input_column,
            labelColumns=self.label_column,
            lookupReference=self.validate_set,
        )
        self.lookUps = self.test_set.lookup_tables

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
