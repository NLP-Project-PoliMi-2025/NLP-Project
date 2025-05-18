from argparse import ArgumentParser
from pathlib import Path
from pyargwriter import api
from project.utils.parser import setup_entrypoint_parser
from project.entrypoint import EntryPoint
from project.utils.parser import setup_parser


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
