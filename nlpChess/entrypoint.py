class EntryPoint:
    def train_next_token(
        self,
        model_type: str,
        max_epochs: int = 10,
        lr: float = 1e-3,
        checkpoint_dir="checkpoints",
    ):
        from nlpChess.scripts.train_next_token import train

        train(model_type, max_epochs, lr, checkpoint_dir)

    def train_game_annotation(
        sef,
        label: str,
        model_type: str = "rnn",
        max_epochs: int = 10,
        lr: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
        bidirectional: bool = True,
        n_layers: int = 2,
        d_model: int = 512,
        extensive_logging: bool = False,
    ):
        from nlpChess.scripts.train_game_annotation import train

        train(
            label,
            model_type,
            max_epochs,
            lr,
            checkpoint_dir,
            bidirectional,
            n_layers,
            d_model,
            log_last_token_metrics=extensive_logging,
        )

    def start_chess_bot(
        self,
        model_weights: str,
        start_fen: str = None,
        bot_starts: bool = False,
        epsilon_greedy: float = 0,
        use_vocal: bool = False,
    ):
        from nlpChess.ChessPlayerApplet.ChessPlayerAppletMouse import (
            ChessPlayerAppletMouse,
        )
        from nlpChess.ChessPlayerApplet.chess_bot import LSTMChessBot
        import yaml

        with open("data/games_0001/moves_lookup_table.yaml", "r") as f:
            look_up_table = yaml.safe_load(f)
        with open("data/games_0001/result_seqs_lookup_table.yaml", "r") as f:
            look_up_table.update(yaml.safe_load(f))

        chess_bot = LSTMChessBot(
            weight_location=model_weights,
            vocab_table=look_up_table,
            bot_starts=bot_starts,
            epsilon=epsilon_greedy,
        )
        if use_vocal:
            from nlpChess.ChessPlayerApplet.ChessPlayerAppletVocal import (
                ChessPlayerAppletVocal,
            )

            applet = ChessPlayerAppletVocal(
                fen=start_fen, botActionFunction=chess_bot)

        else:
            from nlpChess.ChessPlayerApplet.ChessPlayerAppletMouse import (
                ChessPlayerAppletMouse,
            )

            applet = ChessPlayerAppletMouse(
                fen=start_fen, botActionFunction=chess_bot)

        applet.run()
