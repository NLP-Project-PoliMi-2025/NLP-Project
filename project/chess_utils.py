
from typing import List

import chess


def get_board_states(uci_game: List[str]) -> List[str]:
    board = chess.Board()
    board.reset()
    board_states = [board.fen()]
    for move in uci_game:
        move = chess.Move.from_uci(move)
        board.push(move)
        board_states.append(board.fen())
    return board_states