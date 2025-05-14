from typing import Tuple
from lightning.pytorch import LightningModule
from torch import Tensor, nn


class BoardFenPredictor(LightningModule):
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
