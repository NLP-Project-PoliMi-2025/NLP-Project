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

    def train_game_annotation(
        sef,
        label: str,
        model_type: str = "rnn",
        max_epochs: int = 10,
        lr: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
    ):
        from project.scripts.train_game_annotation import train

        train(label, model_type, max_epochs, lr, checkpoint_dir)

    def start_chess_bot(self, model_weights: str, start_fen: str = None):
        from project.ChessPlayerApplet.ChessPlayerApplet import ChessPlayerApplet

        applet = ChessPlayerApplet(fen=start_fen)
        applet.run()
