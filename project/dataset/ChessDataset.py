import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
import tqdm


class ChessDataset(Dataset):
    def __init__(self, parquette_path: str, inputColumns: List[str], labelColumns: List[str]):
        self.parquette_path = parquette_path
        self.inputColums = inputColumns
        self.labelColumns = labelColumns

        self.__load_parquet()
        self.__convertToIndices()

    def __load_parquet(self):
        print(f"Loading parquet file @ ", self.parquette_path, " with columns ",
              self.inputColums + self.labelColumns)
        # Load the parquet file
        self.df = pd.read_parquet(
            self.parquette_path, columns=self.inputColums + self.labelColumns)

        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

    def __convertToIndices(self):
        # For each column of the df build a lookup table
        self.lookup_tables = {}

        for column in tqdm.tqdm(self.df.columns, desc="Building lookup tables"):
            # Check if the type of the data is a list
            if self.df[column].dtype == "O" and self.df[column].apply(lambda x: isinstance(x, list)).any():
                # Explode the column
                explodedColum = self.df[column].explode()

                # Get the unique values in the column
                unique_values = explodedColum.unique()
            else:
                # Get the unique values in the column
                unique_values = self.df[column].unique()

            # Create a lookup table
            lookup_table = {value: idx for idx,
                            value in enumerate(unique_values)}

            # Add the lookup table to the dictionary
            self.lookup_tables[column] = lookup_table

        # Use the lookups to convert the columns to indices
        for column in tqdm.tqdm(self.df.columns, desc="Converting columns to indices"):
            if self.df[column].dtype == "O" and self.df[column].apply(lambda x: isinstance(x, list)).any():
                # Convert the column to indices
                self.df[column] = self.df[column].map(
                    lambda x: [self.lookup_tables[column][i] for i in x])
            else:
                self.df[column] = self.df[column].map(
                    self.lookup_tables[column]).astype("int32")

    def __getitem__(self, index):
        """
        Args:
            index (int): game index
        """
        # Get the row at the given index
        row = self.df.iloc[index]

        # Get the input and label columns
        inputs = row[self.inputColums].values
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
