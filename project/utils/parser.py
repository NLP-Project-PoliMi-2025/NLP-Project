from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_start_chess_bot_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-weights",
        help="--no-documentation-exists--",
        dest="model_weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start-fen",
        help="--no-documentation-exists--",
        dest="start_fen",
        type=str,
        default=None,
        required=False,
    )
    return parser


def add_train_game_annotation_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--label",
        help="--no-documentation-exists--",
        dest="label",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-type",
        help="--no-documentation-exists--",
        dest="model_type",
        default="rnn",
        required=False,
    )
    parser.add_argument(
        "--max-epochs",
        help="--no-documentation-exists--",
        dest="max_epochs",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--lr",
        help="--no-documentation-exists--",
        dest="lr",
        default=0.001,
        required=False,
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="--no-documentation-exists--",
        dest="checkpoint_dir",
        default="checkpoints",
        required=False,
    )
    return parser


def add_train_next_token_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-type",
        help="--no-documentation-exists--",
        dest="model_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-epochs",
        help="--no-documentation-exists--",
        dest="max_epochs",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--lr",
        help="--no-documentation-exists--",
        dest="lr",
        type=float,
        default=0.001,
        required=False,
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="--no-documentation-exists--",
        dest="checkpoint_dir",
        default="checkpoints",
        required=False,
    )
    return parser


def setup_entrypoint_parser(
    parser: ArgumentParser,
) -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    subparser = {}
    command_subparser = parser.add_subparsers(dest="command", title="command")
    train_next_token = command_subparser.add_parser(
        "train-next-token", help="--no-documentation-exists--"
    )
    train_next_token = add_train_next_token_args(train_next_token)
    subparser["train_next_token"] = train_next_token
    train_game_annotation = command_subparser.add_parser(
        "train-game-annotation", help="--no-documentation-exists--"
    )
    train_game_annotation = add_train_game_annotation_args(train_game_annotation)
    subparser["train_game_annotation"] = train_game_annotation
    start_chess_bot = command_subparser.add_parser(
        "start-chess-bot", help="--no-documentation-exists--"
    )
    start_chess_bot = add_start_chess_bot_args(start_chess_bot)
    subparser["start_chess_bot"] = start_chess_bot
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
