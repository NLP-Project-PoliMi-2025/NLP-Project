# TODO: new w2v embedding table from Paolo and Marti

from typing import List, Tuple
from torch import Tensor, nn
import torch
import pytorch_lightning as pl
from gensim.models import Word2Vec

from project.models.miniGRU import MinimalGRU
from project.models.position_encoding import SinusoidalPositionalEmbedding
from project.utils.arithmetic import get_last_nonzero_indices, pad_last_dim

# metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


class BoardFenPredictor(pl.LightningModule):
    def __init__(
        self,
        encoder_network: nn.Module,
        decoder_network: nn.Module,
        n_epochs: int,
        teacher_forcing: float = 0.0,  # 0 means every recurrent input comes from the model
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_epochs = n_epochs
        self.encoder = encoder_network
        self.decoder = decoder_network

    def _encode_moves(self, moves: Tensor) -> Tensor:
        # get last hidden state
        _, hidden = self.encoder.forward(moves)
        return hidden[:, -1]

    def _decode_board_fen(self, hidden_state: Tensor, targets: Tensor) -> Tensor:
        self.decoder.forward()

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        x, y = batch
        last_hidden = self._encode_moves(moves=x)

        # based on last hidden state predict the new tokens

        return super().training_step()

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        return super().validation_step()

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        return super().test_step()

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Calculate total steps (total_batches = steps per epoch * number of epochs)
        total_steps = self.trainer.estimated_stepping_batches

        # Define the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # Peak learning rate
            total_steps=total_steps,  # Total number of steps
            pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
            anneal_strategy="linear",  # Annealing strategy: 'linear' or 'cos'
            div_factor=25.0,  # Initial learning rate = max_lr / div_factor
            final_div_factor=1e4,  # Final learning rate = max_lr / final_div_factor
        )

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the learning rate after every step
                "frequency": 1,  # How often to call the scheduler
            },
        }


class NextTokenPredictor(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=1e-3,
        model_type="rnn",  # 'rnn' or 'transformer'
        word2vec: str = None,
        freeze_embeddings: bool = False,
        hidden_lmbda: float = 0.0,  # Regularization parameter
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_lmbda = hidden_lmbda  # Regularization parameter
        self.weight_decay = weight_decay

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
            self.embedding = nn.Embedding(vocab_size, d_model)

        if model_type == "rnn":
            self.rnn = nn.LSTM(
                input_size=self.embedding.embedding_dim,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True,
            )
        elif model_type == "transformer":
            self.dim_map = nn.Linear(self.embedding.embedding_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, dim_feedforward=256
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        elif model_type == "mini-gru":
            self.rnn = MinimalGRU(
                input_dim=self.embedding.embedding_dim,
                hidden_dim=d_model,
                output_dim=d_model,
            )

        else:
            raise ValueError("model_type must be 'rnn' or 'transformer' or 'mini-gru")

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab_size - 1)
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        embedded = self.embedding(x)  # (batch, seq_len, d_model)

        if self.model_type in ["rnn", "mini-gru"]:
            output, hidden = self.rnn(embedded)
        else:  # Transformer
            # Transformer expects (seq_len, batch, d_model)
            embedded = self.dim_map.forward(embedded)
            embedded = embedded.permute(1, 0, 2)
            output = self.transformer(embedded)
            # back to (batch, seq_len, d_model)
            output = output.permute(1, 0, 2)
            hidden = None
        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch  # y is the next token for each x_t
        logits, hidden = self(x)
        logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        y = y[:, 1:].contiguous().view(-1)
        loss = self.loss_fn(logits, y)

        if self.model_type == "mini-gru":
            # do hidden state regularization
            reg_loss = self.hidden_lmbda * self.reg_loss_fn(
                hidden, torch.zeros_like(hidden)
            )
            loss = loss + reg_loss
            self.log("hidden_reg_loss", reg_loss)

        # do weight decay
        trainable_params = torch.cat(
            [p.flatten() for p in self.parameters() if p.requires_grad]
        )
        weight_decay_loss = torch.linalg.norm(trainable_params, 2) * self.weight_decay

        loss = loss + weight_decay_loss

        self.log("max_weight", trainable_params.max())
        self.log("train_weight_decay_loss", weight_decay_loss)
        self.log("train_loss", loss)

        self.get_metrics(logits, y, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        y = y[:, 1:].contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        self.get_metrics(logits, y, "val")
        return loss

    def get_metrics(self, logits: torch.Tensor, y: torch.Tensor, stage: str):
        # logits: (batch_size, seq_len, vocab_size)
        # y: (batch_size, seq_len)

        preds = torch.argmax(logits, dim=-1)

        # flatten the tensors
        preds = preds.view(-1).cpu().numpy()
        y = y.view(-1).cpu().numpy()

        # do accuracy for multi-class classification
        acc = accuracy_score(y, preds)
        self.log(f"{stage}_acc", acc)
        # do precision for multi-class classification
        precision = precision_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}_precision", precision)
        # do recall for multi-class classification
        recall = recall_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}_recall", recall)
        # do f1 for multi-class classification
        f1 = f1_score(y, preds, average="weighted")
        self.log(f"{stage}_f1", f1)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Calculate total steps (total_batches = steps per epoch * number of epochs)
        total_steps = self.trainer.estimated_stepping_batches

        # Define the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # Peak learning rate
            total_steps=total_steps,  # Total number of steps
            pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
            anneal_strategy="linear",  # Annealing strategy: 'linear' or 'cos'
            div_factor=25.0,  # Initial learning rate = max_lr / div_factor
            final_div_factor=1e4,  # Final learning rate = max_lr / final_div_factor
        )

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the learning rate after every step
                "frequency": 1,  # How often to call the scheduler
            },
        }


class SeqAnnotator(pl.LightningModule):
    def __init__(
        self,
        n_target_classes: int,
        label: str,
        vocab_size: int,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=1e-3,
        model_type="lstm",  # 'lstm' or 'transformer'
        ignore_index: int = None,
        word2vec: str = None,
        freeze_embeddings: bool = False,
        label_counts: pd.DataFrame = None,
        bidirectional: bool = False,
        logging_last_token_metrics: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.n_target_classes = n_target_classes
        self.label = label
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.model_type = model_type
        self.ignore_index = ignore_index
        self.word2vec = word2vec
        self.freeze_embeddings = freeze_embeddings
        self.bidirectional = bidirectional
        self.logging_last_token_metrics = logging_last_token_metrics

        self.embedding: nn.Embedding
        self.rnn: nn.Module
        self.transformer: nn.Module
        self.fc_out: nn.Module
        self.build_model()

        if label_counts is not None:
            loss_weights = 1 / label_counts
            loss_weights = loss_weights / loss_weights.sum()
            loss_weights = loss_weights.sort_index().values
            loss_weights = np.concatenate([np.zeros(1), loss_weights])
        else:
            loss_weights = np.ones(self.n_target_classes + 1)

        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.from_numpy(loss_weights).float(),
            ignore_index=self.ignore_index,
        )
        self.reg_loss_fn = nn.MSELoss()

        self.example_input_array = torch.randint(
            0, self.vocab_size, (1, 10), dtype=torch.long
        ).to(self.device)

    def build_model(self):
        if self.word2vec is not None:
            # Assume w2v_model is a gensim Word2Vec model
            w2v_model = Word2Vec.load(self.word2vec)
            embedding_matrix = torch.FloatTensor(
                w2v_model.wv.vectors
            )  # shape: (vocab_size, embedding_dim)

            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=self.freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        core_out_dim = self.d_model
        if self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.embedding.embedding_dim,
                hidden_size=self.d_model,
                num_layers=self.n_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.embedding.embedding_dim,
                hidden_size=self.d_model,
                num_layers=self.n_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        elif self.model_type == "transformer":
            self.positional_encoding = SinusoidalPositionalEmbedding(
                self.embedding.embedding_dim
            )
            self.dim_map = nn.Linear(self.embedding.embedding_dim, self.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dropout=self.dropout,
                dim_feedforward=256,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=self.n_layers
            )
        elif self.model_type == "mini-gru":
            self.rnn = MinimalGRU(
                input_dim=self.embedding.embedding_dim,
                hidden_dim=self.d_model,
                output_dim=self.d_model,
            )
        else:
            raise ValueError("model_type must be 'lstm' or 'transformer' or 'mini-gru")

        if self.bidirectional:
            core_out_dim = self.d_model * 2

        self.fc_out = nn.Sequential(
            nn.Linear(core_out_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.n_target_classes),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, x: Tensor, logits: bool = False, hx: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len)
        """
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        if self.model_type in ["gru", "lstm", "mini-gru"]:
            output, hidden = self.rnn.forward(
                embedded,
            )
        else:  # Transformer
            # add positional encoding
            seq_len = embedded.size(1)
            embedded = (
                self.positional_encoding.forward(seq_len).to(embedded.device) + embedded
            )
            # Transformer expects (seq_len, batch, d_model)
            embedded = self.dim_map.forward(embedded)
            embedded = embedded.permute(1, 0, 2)
            output = self.transformer(embedded)
            # back to (batch, seq_len, d_model)
            output = output.permute(1, 0, 2)
            hidden = None

        if logits:
            # Adjust index based on desired layer
            partial_model = nn.Sequential(*list(self.fc_out.children())[:-1])
            output = partial_model(output)
        else:
            output = self.fc_out(output)  # (batch, seq_len, vocab_size)

        return output, hidden

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        # (batch_size, seq_length, n_target_classes)
        logits, _ = self.forward(x)
        logits = pad_last_dim(logits, pad_value=0)
        self.get_last_token_metrics(logits, y, "train")

        logits = logits.contiguous().view(-1, self.n_target_classes + 1)
        y = y.contiguous().view(-1)

        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss)

        self.get_metrics(logits, y, "train")

        # log learning rate
        for param_group in self.optimizers().param_groups:
            self.log("lr", param_group["lr"])

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        # (batch_size, seq_length, n_target_classes)
        logits, _ = self.forward(x)
        logits = pad_last_dim(logits, pad_value=0)
        self.get_last_token_metrics(logits, y, "val")

        logits = logits.contiguous().view(-1, self.n_target_classes + 1)
        y = y.contiguous().view(-1)

        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss)

        self.get_metrics(logits, y, "val")
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        # (batch_size, seq_length, n_target_classes)
        logits, _ = self.forward(x)
        logits = pad_last_dim(logits, pad_value=0)
        self.get_last_token_metrics(logits, y, "test")

        logits = logits.contiguous().view(-1, self.n_target_classes + 1)
        y = y.contiguous().view(-1)

        loss = self.loss_fn(logits, y)
        self.log("test/loss", loss)

        self.get_metrics(logits, y, "test")

        return loss

    def get_metrics(self, logits: torch.Tensor, y: torch.Tensor, stage: str):
        # logits: (batch_size, seq_len, vocab_size)
        # y: (batch_size, seq_len)

        preds = torch.argmax(logits, dim=-1)

        # flatten the tensors
        preds = preds.view(-1).cpu().numpy()
        y = y.view(-1).cpu().numpy()

        # do accuracy for multi-class classification
        acc = accuracy_score(y, preds)
        self.log(f"{stage}/acc", acc)
        # do precision for multi-class classification
        precision = precision_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}/precision", precision)
        # do recall for multi-class classification
        recall = recall_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}/recall", recall)
        # do f1 for multi-class classification
        f1 = f1_score(y, preds, average="weighted")
        self.log(f"{stage}/f1", f1)

    def get_last_token_metrics(self, logits: torch.Tensor, y: torch.Tensor, stage: str):
        if not self.logging_last_token_metrics or self.label not in [
            "termination_seqs",
            "result_seqs",
        ]:
            return

        preds = torch.argmax(logits, dim=-1)

        # compute metrics only for the last token
        indices = get_last_nonzero_indices(y)
        if len(y.shape) == 1:
            indices = indices.unsqueeze(0)
            preds = preds.unsqueeze(0)

        batch_size = indices.shape[0]
        y = y[torch.arange(batch_size), indices].cpu().numpy()
        preds = preds[torch.arange(batch_size), indices].cpu().numpy()

        # do accuracy for multi-class classification
        acc = accuracy_score(y, preds)
        self.log(f"{stage}/acc_last", acc)
        # do precision for multi-class classification
        precision = precision_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}/precision_last", precision)
        # do recall for multi-class classification
        recall = recall_score(
            y,
            preds,
            average="weighted",
            zero_division=0,
        )
        self.log(f"{stage}/recall_last", recall)
        # do f1 for multi-class classification
        f1 = f1_score(y, preds, average="weighted")
        self.log(f"{stage}/f1_last", f1)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Calculate total steps (total_batches = steps per epoch * number of epochs)
        total_steps = self.trainer.estimated_stepping_batches

        # Define the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # Peak learning rate
            total_steps=total_steps,  # Total number of steps
            pct_start=0.3,  # Percentage of the cycle spent increasing the learning rate
            anneal_strategy="cos",  # Annealing strategy: 'linear' or 'cos'
            div_factor=25.0,  # Initial learning rate = max_lr / div_factor
            final_div_factor=1e4,  # Final learning rate = max_lr / final_div_factor
        )

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the learning rate after every step
                "frequency": 1,  # How often to call the scheduler
            },
        }
