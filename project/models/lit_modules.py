# TODO: new w2v embedding table from Paolo and Marti

from typing import List, Tuple
from torch import Tensor, nn
import torch
import pytorch_lightning as pl
from gensim.models import Word2Vec

from project.models.miniGRU import MinimalGRU

# metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        return super().configure_optimizers()


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
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class SeqAnnotator(pl.LightningModule):
    def __init__(
        self,
        n_target_classes: int,
        vocab_size: int,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=1e-3,
        model_type="rnn",  # 'rnn' or 'transformer'
        ignore_index: int = None,
        word2vec: str = None,
        freeze_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_target_classes = n_target_classes
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

        self.embedding: nn.Embedding
        self.rnn: nn.Module
        self.transformer: nn.Module
        self.fc_out: nn.Module
        self.build_model()
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.reg_loss_fn = nn.MSELoss()

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

        
        if self.model_type == "rnn":
            self.rnn = nn.LSTM(
                input_size=self.embedding.embedding_dim,
                hidden_size=self.d_model,
                num_layers=self.n_layers,
                batch_first=True,
            )
        elif self.model_type == "transformer":
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
            raise ValueError("model_type must be 'rnn' or 'transformer' or 'mini-gru")

        self.fc_out = nn.Linear(self.d_model, self.n_target_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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

    def training_step(
        self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx, dataloader_idx
    ):
        x, y = batch
        x = x[0]
        y = y[0]
        logits, _ = self.forward(x)  # (batch_size, seq_length, n_target_classes)
        logits = logits.contiguous().view(-1, self.n_target_classes)
        y = y.contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        self.get_metrics(logits, y, "train")
        return loss

    def validation_step(
        self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx, dataloader_idx
    ):
        x, y = batch
        x = x[0]
        y = y[0]
        logits, _ = self.forward(x)  # (batch_size, seq_length, n_target_classes)
        logits = logits.contiguous().view(-1, self.n_target_classes)
        y = y.contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

        self.get_metrics(logits, y, "val")
        return loss

    def test_step(self, batch: Tuple[List[Tensor], List[Tensor]], batch_idx, dataloader_idx):
        x, y = batch
        x = x[0]
        y = y[0]
        logits, _ = self.forward(x)  # (batch_size, seq_length, n_target_classes)
        logits = logits.contiguous().view(-1, self.n_target_classes)
        y = y.contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)

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
