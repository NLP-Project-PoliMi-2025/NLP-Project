from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from nlpChess.models.lit_modules import (
    SeqAnnotator,
)  # or wherever the model is defined
from nlpChess.dataset.lit_module import SeqAnnotationDM
import torch
from datetime import datetime


def train(
    label: str,
    model_type: str = "rnn",
    max_epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    bidirectional: bool = True,
    num_layers: int = 2,
    d_model: int = 512,
    log_last_token_metrics: bool = False,
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
        label=label,
        vocab_size=dm.get_vocab_size()[0],
        d_model=d_model,
        n_layers=num_layers,
        n_heads=4,
        dropout=0.1,
        lr=lr,
        model_type=model_type,
        ignore_index=0,
        freeze_embeddings=True,
        word2vec="model_weights/word2vec.model",
        label_counts=dm.fit_set.df[label].explode().value_counts(),
        bidirectional=bidirectional,
        logging_last_token_metrics=log_last_token_metrics,
    )
    print(dm.get_vocab_size())
    print(model)

    # Checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        dirpath=checkpoint_dir + "/" + label,
        filename=f"{model_type}-best",
    )

    csv_logger = CSVLogger(f"csv_logs/{label}", name=model_type)
    tb_logger = TensorBoardLogger(f"tb_logs/{label}", name=model_type)
    wb_logger = WandbLogger(
        project="NLP-chess",
        name=f"{model_type}-{label}-{datetime.now().strftime(" % Y % m % d % H % M % S")}",
        save_dir=f"wandb_logs/{label}/{model_type}",
        log_model=False,
    )
    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tb_logger, wb_logger],
        accelerator="auto",
        devices="auto",
    )

    # Fit model
    trainer.fit(model, dm)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    # Test best model
    model = SeqAnnotator.load_from_checkpoint(
        checkpoint_callback.best_model_path,
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
        word2vec="model_weights/word2vec.model",
    )
    model.eval()
    test_results = trainer.test(model, dataloaders=dm.test_dataloader())
    print(f"Test results: {test_results}")
    return model
