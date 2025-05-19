from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from project.models.lit_modules import (
    SeqAnnotator,
)  # or wherever the model is defined
from project.dataset.lit_module import NextTokenDM, SeqAnnotationDM
import torch


def train(
    label: str, model_type="rnn", max_epochs=10, lr=1e-3, checkpoint_dir="checkpoints"
):
    torch.set_float32_matmul_precision("medium")
    # Get datamodule
    dm = SeqAnnotationDM(
        "data/games_0001/train_100K.parquet",
        "data/games_0001/val_100K.parquet",
        "data/games_0001/test_100K.parquet",
        32,
        ["Moves"],
        [label],
        8,
    )
    dm.setup()

    # Instantiate model
    model = SeqAnnotator(
        n_target_classes=dm.get_num_labels()[0],
        vocab_size=dm.get_vocab_size()[0],
        d_model=512,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=lr,
        model_type=model_type,
        ignore_index=0,
        freeze_embeddings=True,
        word2vec="model_weights/word2vec.model"
    )
    print(dm.get_vocab_size())
    print(model)

    # Checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=checkpoint_dir,
        filename=f"{model_type}-best",
    )

    csv_logger = CSVLogger("csv_logs", name=model_type)
    tb_logger = TensorBoardLogger("tb_logs", name=model_type)
    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tb_logger],
        accelerator="auto",
        devices="auto",
    )

    # Fit model
    trainer.fit(model, dm)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return model
