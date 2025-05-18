import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import numpy as np


class BiLSTM(nn.Module):
    """
    A Bidirectional LSTM model for sequence classification.
    The model consists of a bidirectional LSTM layer followed by a fully connected layer.
    The LSTM processes the input sequence in both forward and backward directions,
    and the final output is obtained by concatenating the outputs from both directions.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        predictionType: bool,
        num_outcomes,
        word2vec=None,
        freeze_embeddings=False,
        lr=1e-3,
    ):
        """
        Args:
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Number of hidden units in the LSTM layer.
            num_outcomes (int): Number of output classes for classification.
            vocab_size (int): Size of the vocabulary.
            predictionType (bool): If True, the model is used for sequence classification; otherwise, for token classification.
            word2vec (str, optional): Path to a pre-trained Word2Vec model. Defaults to None.
            freeze_embeddings (bool, optional): If True, freeze the word embeddings during training. Defaults to False.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        super(BiLSTM, self).__init__()
        # Bidirectional LSTM; we set batch_first=True to have input like [batch, seq_len, embedding_dim]
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        # Fully connected layer to map hidden state coming from LSTM to output labels
        # (the hidden state is a concatenation of two LSTM outputs since it is bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_outcomes)
        self.predictionType = predictionType
        self.lr = lr
        if word2vec is not None:
            # Assume w2v_model is a gensim Word2Vec model
            w2v_model = Word2Vec.load(word2vec)
            embedding_matrix = torch.FloatTensor(
                w2v_model.wv.vectors
            )  # shape: (vocab_size, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, num_outcomes) for token classification
                          or (batch_size, num_outcomes) for sequence classification.
        """
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # embedded: [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embeds)  # lstm_out: [batch_size, seq_len, hidden_dim*2]
        if self.predictionType:
            # For sequence classification, we take the last output of the LSTM
            lstm_out = lstm_out[:, -1, :]
            logits = self.fc(lstm_out)
        else:
            logits = self.fc(lstm_out)  # logits: [batch_size, seq_len, num_tags]
        return logits

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        Returns:
            torch.optim.Optimizer: The optimizer to be used for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
    def training_step(self, batch):
        """
        Perform a single training step.
        Args:
            batch (tuple): A batch of data containing input and target tensors.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        x, y, lengths = batch
        logits = self(x)
        if self.predictionType:
            # Sequence classification
            logits = self.fc(logits)  # [batch_size, num_outcomes]
            loss = self.loss_fn(logits, y)
        else:
            # Token classification
            logits = self.fc(logits)  # [batch_size, seq_len, num_outcomes]
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch):
        """
        Perform a single validation step.
        Args:
            batch (tuple): A batch of data containing input and target tensors.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        x, y, lengths = batch
        logits = self(x)
        if self.predictionType:
            logits = self.fc(logits)
            loss = self.loss_fn(logits, y)
        else:
            logits = self.fc(logits)
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            loss = self.loss_fn(logits, y)
        return loss

    def test_step(self, batch):
        """
        Perform a single test step.
        Args:
            batch (tuple): A batch of data containing input and target tensors.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        x, y, lengths = batch
        logits = self(x)
        if self.predictionType:
            logits = self.fc(logits)
            loss = self.loss_fn(logits, y)
        else:
            logits = self.fc(logits)
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            loss = self.loss_fn(logits, y)
        return loss

    def set_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss()

    # def collate_fn(self, batch):
    #     """
    #     Custom collate function to handle padding and batching of sequences.
    #     Args:
    #         batch (list): List of tuples, where each tuple contains a word tensor and a tag tensor.
    #     Returns:

    #         torch.Tensor: Padded word tensor of shape (batch_size, max_seq_len, embedding_dim).
    #         torch.Tensor: Padded tag tensor of shape (batch_size, max_seq_len).
    #         list: List of lengths of each sequence in the batch.
    #     """
    #     # batch: list of tuples (word_tensor, tag_tensor)
    #     # Get input sentences
    #     moves = [item[0] for item in batch]
    #     # Get labels
    #     outcomes = [item[1] for item in batch]
    #     # Get maximum length in the batch
    #     lengths = [len(s) for s in moves]
    #     max_len = max(lengths)

    #     # Pad shorter sentences to let the input tensors all have the same size
    #     padded_moves = []
    #     for s in moves:
    #         pad_len = max_len - len(s)
    #         # Padding uses index 0 both for words and labels
    #         padded_moves.append(torch.cat([s, torch.zeros((pad_len, s.shape[1]), dtype=torch.float32)]))

    #     #
    #     return torch.stack(padded_moves), torch.stack(outcomes), lengths
