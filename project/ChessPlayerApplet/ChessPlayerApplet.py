import pygame
from abc import ABC, abstractmethod
import chess
import chess.svg
import cairosvg
import io
from typing import List


class ChessPlayerApplet(ABC):
    """ChessPlayerApplet is a class that creates a chess applet using Pygame and the python-chess library.
    It allows the user to play chess against a bot or another player. The applet displays the chessboard and pieces,
    handles user input, and updates the board state.
    Args:
        board_size (int, optional): The pixel size of the board. Defaults to 800.
        fen (str, optional): The start board configuration, if None the defult is picked. Defaults to None.
        botActionFucntion (function: (List[str], List[str]) -> str, optional): function that takes a list of the performed actions so far in UCI and a list of available moves to get the available moves. If None the user input are taken as bot actions. Defaults to None.
    """

    def __init__(self, board_size=350, fen=None, botActionFunction=None):
        pygame.init()
        self.UCImoves = []
        self.board_size = board_size
        self.padding_x = 14 / 350 * self.board_size
        self.padding_y = 14 / 350 * self.board_size
        self.square_size_x = (self.board_size - 2 * self.padding_x) / 8
        self.square_size_y = (self.board_size - 2 * self.padding_y) / 8
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))
        self.clock = pygame.time.Clock()
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.current_start = None
        self.botActionFunction = botActionFunction

    def performAction(self, UCIMove: chess.Move):
        """Performs a move on the chess board using UCI format.
        This method is used to perform a move on the chess board.

        Args:
            UCIMove (str): The move in UCI format (e.g., "e2e4").

        Raises:
            ValueError: If the move is illegal.
        """
        # Perform a move using UCI format
        move = UCIMove
        if move in self.getLegalMoves():
            self.board.push(move)
            self.render_board()
            self.UCImoves.append(UCIMove.uci())
        else:
            raise ValueError(f"Illegal move: {UCIMove}")

    def getLegalMoves(self) -> List[str]:
        """Returns a list of all legal moves in UCI format.
        This method is used to get the legal moves for the current position.

        Returns:
            List[str]: A list of legal moves in UCI format.
        """
        # Get all legal moves in UCI format
        return [move for move in self.board.legal_moves]

    def render_board(self, start=None):
        """Renders the chess board using Pygame and the python-chess library.
        This method is used to render the chess board and pieces on the screen.
        Args:
            start (str, optional): The starting square of the move in UCI format. Defaults to None.
        """
        # Render board using chess library
        if start is not None:
            from_square = chess.parse_square(start)
            possible_moves = [
                move.to_square
                for move in self.board.legal_moves
                if move.from_square == from_square
            ]
            fill = {from_square: "#1E90FF"}
            for sq in possible_moves:
                fill[sq] = "#32CD3280"
            board_svg = chess.svg.board(self.board, size=self.board_size, fill=fill)
        else:
            board_svg = chess.svg.board(self.board, size=self.board_size)

        board_png = cairosvg.svg2png(bytestring=board_svg)
        background = pygame.image.load(io.BytesIO(board_png)).convert_alpha()
        self.screen.blit(background, (0, 0))
        pygame.display.flip()

    @abstractmethod
    def run(self):
        """Runs the main loop of the applet."""
        pass
