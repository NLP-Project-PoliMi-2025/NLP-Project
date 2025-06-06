from argparse import ArgumentParser
from pathlib import Path
from pyargwriter import api
from nlpChess.utils.parser import setup_entrypoint_parser
from nlpChess.entrypoint import EntryPoint
from nlpChess.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = EntryPoint()
    _, command_parser = setup_entrypoint_parser(ArgumentParser())
    match args["command"]:
        case "train-next-token":
            module.train_next_token(
                model_type=args["model_type"],
                max_epochs=args["max_epochs"],
                lr=args["lr"],
                checkpoint_dir=args["checkpoint_dir"],
            )

        case "train-game-annotation":
            module.train_game_annotation(
                label=args["label"],
                model_type=args["model_type"],
                max_epochs=args["max_epochs"],
                lr=args["lr"],
                checkpoint_dir=args["checkpoint_dir"],
                bidirectional=args["bidirectional"],
                n_layers=args["n_layers"],
                d_model=args["d_model"],
                extensive_logging=args["extensive_logging"],
            )

        case "start-chess-bot":
            module.start_chess_bot(
                model_weights=args["model_weights"],
                start_fen=args["start_fen"],
                bot_starts=args["bot_starts"],
                epsilon_greedy=args["epsilon_greedy"],
                use_vocal=args["use_vocal"],
            )

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="--no-documentation-exists--")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
