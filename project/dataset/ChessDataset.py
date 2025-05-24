import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
import tqdm
import numpy as np
from pyarrow.lib import ArrowInvalid
from datasets import Dataset as huggingfaceDataset


class ChessDataset(Dataset):
    def __init__(
        self,
        inputColumns: List[str],
        labelColumns: List[str],
        dataset: huggingfaceDataset = None,
        parquette_path: str = "",
        lookupReference: "ChessDataset" = None,
    ):
        self.parquette_path = parquette_path
        assert (
            lookupReference is None or lookupReference.lookup_tables is not None
        ), "Lookup tables reference has no lookup tables"
        self.lookupReference = lookupReference
        self.inputColumns = inputColumns
        self.labelColumns = labelColumns
        self.dataset: huggingfaceDataset = dataset

        self.__instantiateLookup()
        self.__load_parquet() if self.dataset is None else self.__load_from_dataset()
        if self.df is None:
            raise ValueError("DataFrame is None, cannot load data")
        self.__convertToIndices()

    def __load_from_dataset(self):
        print(
            f"Loading dataset with columns ",
            self.inputColumns + self.labelColumns,
        )
        if self.dataset is None:
            raise ValueError("Dataset is None, cannot load data")

        # Convert the dataset to a pandas DataFrame
        self.df = self.dataset.to_pandas(
        )[self.inputColumns + self.labelColumns]

        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

    def __load_parquet(self):
        print(
            f"Loading parquet file @ ",
            self.parquette_path,
            " with columns ",
            self.inputColumns + self.labelColumns,
        )
        try:
            # Load the parquet file
            self.df = pd.read_parquet(
                self.parquette_path, columns=self.inputColumns + self.labelColumns
            )
        except ArrowInvalid as e:
            raise ValueError(
                "columns: ", self.inputColumns + self.labelColumns, " not found"
            ) from e
        except FileNotFoundError as e:
            raise ValueError("File not found: ", self.parquette_path) from e
        except Exception as e:
            raise ValueError("Error loading parquet file: ",
                             self.parquette_path) from e

        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

    def __instantiateLookup(self):
        if self.lookupReference is None:
            self.lookup_tables = {}
            return

        self.lookup_tables = self.lookupReference.lookup_tables

    def __checkToBeExploded(self, column):
        """
        Checks if a specified DataFrame column should be "exploded" based on its data type and contents.

        A column is considered to be exploded if:
        - Its dtype is 'object' (typically string or mixed types in pandas), and
        - It contains at least one element that is a list or a NumPy ndarray.

        Args:
            column (str): The name of the column to check in the DataFrame.

        Returns:
            bool: True if the column should be exploded (i.e., contains lists or ndarrays), False otherwise.
        """
        return self.df[column].dtype == "O" and (
            self.df[column].apply(lambda x: isinstance(x, list)).any()
            or self.df[column].apply(lambda x: isinstance(x, np.ndarray)).any()
        )

    def __convertToIndices(self):
        for column in tqdm.tqdm(self.df.columns, desc="Building lookup tables"):
            # Check if the type of the data is a list
            if self.__checkToBeExploded(column):
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

            maxId = max(lookup_table.values()) if lookup_table else 1

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
        print(index)
        # Get the row at the given index
        row = self.df.iloc[index]

        # Get the input and label columns
        inputs = row[self.inputColumns]
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
