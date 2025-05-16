from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from project.models.lit_modules import (
    NextTokenPredictor,
)  # or wherever the model is defined
from project.dataset.lit_module import NextTokenDM


def train(model_type="rnn", max_epochs=10, lr=1e-3, checkpoint_dir="checkpoints"):
    # Get datamodule
    dm = NextTokenDM("data/chess_games_1.db", num_worker=8, batch_size=1, use_ram=True)

    # Instantiate model
    encoder_weight_location = "model_weights/word2vec.model"
    model = NextTokenPredictor(
        vocab_size=dm.get_vocab_size(),
        model_type=model_type,
        lr=lr,
        word2vec=encoder_weight_location,
        freeze_embeddings=True,
    )

    # Checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=checkpoint_dir,
        filename=f"{model_type}-best",
    )

    logger = CSVLogger("logs", name=model_type)

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
    )

    # Fit model
    trainer.fit(model, dm)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return model


def evaluate(checkpoint_path, vocab_size=1000):
    # Reload the data module
    dm = NextTokenDM("data/chess_games_1.db")

    # Infer model type from checkpoint name
    if "transformer" in checkpoint_path:
        model_type = "transformer"
    else:
        model_type = "rnn"

    # Load model
    model = NextTokenPredictor.load_from_checkpoint(
        checkpoint_path, vocab_size=vocab_size, model_type=model_type
    )

    trainer = Trainer(accelerator="auto", devices="auto")
    results = trainer.validate(model, datamodule=dm)
    return results
