from argparse import ArgumentParser
from typing import Tuple, Dict, List


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
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
