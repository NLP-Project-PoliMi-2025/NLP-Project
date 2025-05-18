import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
import tqdm
import numpy as np


class ChessDataset(Dataset):
    def __init__(
        self, parquette_path: str, inputColumns: List[str], labelColumns: List[str], lookupReference: 'ChessDataset' = None
    ):
        self.parquette_path = parquette_path
        assert lookupReference is None or lookupReference.lookup_tables is not None, "Lookup tables reference has no lookup tables"
        self.lookupReference = lookupReference
        self.inputColumns = inputColumns
        self.labelColumns = labelColumns

        self.__instantiateLookup()
        self.__load_parquet()
        self.__convertToIndices()

    def __load_parquet(self):
        print(
            f"Loading parquet file @ ",
            self.parquette_path,
            " with columns ",
            self.inputColumns + self.labelColumns,
        )
        # Load the parquet file
        self.df = pd.read_parquet(
            self.parquette_path, columns=self.inputColumns + self.labelColumns
        )

        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

    def __instantiateLookup(self):
        if self.lookupReference is None:
            self.lookup_tables = {}
            return

        self.lookup_tables = self.lookupReference.lookup_tables

    def __checkToBeExploded(self, column):
        return self.df[column].dtype == "O" and (
            self.df[column].apply(lambda x: isinstance(x, list)).any() or
            self.df[column].apply(
                lambda x: isinstance(x, np.ndarray)).any()
        )

    def __convertToIndices(self):
        for column in tqdm.tqdm(self.df.columns, desc="Building lookup tables"):
            # Check if the type of the data is a list
            if (self.__checkToBeExploded(column)):
                # Explode the column
                explodedColum = self.df[column].explode()

                # Get the unique values in the column
                unique_values = explodedColum.unique()
            else:
                # Get the unique values in the column
                unique_values = self.df[column].unique()

            if column not in self.lookup_tables:
                # Create a lookup table for the column
                lookup_table = {}
            else:
                # Get the lookup table for the column
                lookup_table = self.lookup_tables[column]

            maxId = max(lookup_table.values()) if lookup_table else 0

            for value in unique_values:
                # Check if the value is already in the lookup table
                if value not in lookup_table:
                    # Add the value to the lookup table
                    lookup_table[value] = maxId
                    maxId += 1

            # Add the lookup table to the dictionary
            self.lookup_tables[column] = lookup_table

        # Use the lookups to convert the columns to indices
        for column in tqdm.tqdm(self.df.columns, desc="Converting columns to indices"):
            if self.__checkToBeExploded(column):
                # Convert the column to indices
                self.df[column] = self.df[column].map(
                    lambda x: [self.lookup_tables[column][i] for i in x]
                )
            else:
                self.df[column] = (
                    self.df[column].map(
                        self.lookup_tables[column]).astype("int32")
                )

    def __getitem__(self, index):
        """
        Args:
            index (int): game index
        """
        # Get the row at the given index
        row = self.df.iloc[index]

        # Get the input and label columns
        inputs = row[self.inputColumns].values
        # Convert to tensors
        inputs = [torch.tensor(i) for i in inputs]

        labels = row[self.labelColumns].values
        labels = [torch.tensor(i) for i in labels]

        return inputs, labels

    def __len__(self):
        return len(self.df)

    def getLookupTable(self, column: str):
        """
        Args:
            column (str): column name
        """
        return self.lookup_tables[column]

    def getInputLookupTables(self):
        """
        Returns:
            dict: dictionary of input lookup tables
        """
        return {column: self.lookup_tables[column] for column in self.inputColumns}

    def getLabelLookupTables(self):
        """
        Returns:
            dict: dictionary of label lookup tables
        """
        return {column: self.lookup_tables[column] for column in self.labelColumns}
