class EntryPoint:
    def train_next_token(
        self,
        model_type: str,
        max_epochs: int = 10,
        lr: float = 1e-3,
        checkpoint_dir="checkpoints",
    ):
        from project.scripts.train_next_token import train

        train(model_type, max_epochs, lr, checkpoint_dir)
