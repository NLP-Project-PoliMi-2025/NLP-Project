from typing import Tuple
from torch import Tensor, nn
import torch
import pytorch_lightning as pl
from gensim.models import Word2Vec


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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr
        self.vocab_size = vocab_size
        self.d_model = d_model

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
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            raise ValueError("model_type must be 'rnn' or 'transformer'")

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        embedded = self.embedding(x)  # (batch, seq_len, d_model)

        if self.model_type == "rnn":
            output, _ = self.rnn(embedded)
        else:  # Transformer
            # Transformer expects (seq_len, batch, d_model)
            embedded = embedded.permute(1, 0, 2)
            output = self.transformer(embedded)
            output = output.permute(1, 0, 2)  # back to (batch, seq_len, d_model)

        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch  # y is the next token for each x_t
        logits = self(x)
        logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        y = y[:, 1:].contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        y = y[:, 1:].contiguous().view(-1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
