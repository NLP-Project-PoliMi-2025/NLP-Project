import pygame
from pygame.locals import *
import chess
import chess.svg
import cairosvg
import io
from typing import List
import random


class ChessPlayerApplet:
    def __init__(self, board_size=800, fen=None, botActionFunction=None):
        """ChessPlayerApplet is a class that creates a chess applet using Pygame and the python-chess library.
        It allows the user to play chess against a bot or another player. The applet displays the chessboard and pieces,
        handles user input, and updates the board state.

        Args:
            board_size (int, optional): The pizel size of the board. Defaults to 800.
            fen (str, optional): The start board configuration, if None the defult is picked. Defaults to None.
            botActionFucntion (function: (List[str], List[str]) -> str, optional): function that takes a list of the performed actions so far in UCI and a list of available moves to get the available moves. If None the user input are taken as bot actions. Defaults to None.
        """
        pygame.init()
        self.UCImoves = []
        self.board_size = board_size
        self.padding_x = 14 / 350 * self.board_size
        self.padding_y = 14 / 350 * self.board_size
        self.square_size_x = (self.board_size - 2 * self.padding_x) / 8
        self.square_size_y = (self.board_size - 2 * self.padding_y) / 8
        self.screen = pygame.display.set_mode(
            (self.board_size, self.board_size))
        self.clock = pygame.time.Clock()
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.current_start = None
        self.botActionFunction = botActionFunction

    def pos2uci(self, pos):
        # Convert pixel position to UCI format
        x, y = pos
        x = int((x - self.padding_x) / self.square_size_x)
        y = int((y - self.padding_y) / self.square_size_y)
        uci = chess.square_name(x + (7 - y) * 8)
        return uci

    def uci2pos(self, uci):
        # Convert UCI format to pixel position
        square = chess.parse_square(uci)
        x = (square % 8) * self.square_size_x + self.padding_x
        y = (7 - (square // 8)) * self.square_size_y + self.padding_y
        return (x, y)

    def render_board(self, start=None):
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
            board_svg = chess.svg.board(
                self.board, size=self.board_size, fill=fill)
        else:
            board_svg = chess.svg.board(self.board, size=self.board_size)

        board_png = cairosvg.svg2png(bytestring=board_svg)
        background = pygame.image.load(io.BytesIO(board_png)).convert_alpha()
        self.screen.blit(background, (0, 0))
        pygame.display.flip()

    def run(self):
        self.render_board()
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == MOUSEBUTTONDOWN:
                    current_pointer = self.pos2uci(event.pos)
                    if self.current_start and current_pointer != self.current_start:
                        move = chess.Move.from_uci(
                            f"{self.current_start}{current_pointer}"
                        )
                        print(
                            f"Trying to move {self.current_start} to {current_pointer}"
                        )
                        if move in self.board.legal_moves:
                            self.performAction(move)

                            if self.botActionFunction is not None:
                                legal_moves = self.getLegalMoves()
                                legal_moves = [move.uci()
                                               for move in legal_moves]
                                self.performAction(
                                    chess.Move.from_uci(
                                        self.botActionFunction(
                                            self.UCImoves, legal_moves
                                        )
                                    )
                                )

                        self.current_start = None
                        self.render_board(self.current_start)
                    else:
                        self.current_start = current_pointer
                        # Check if there are possible moves from the current square
                        possible_moves = [
                            move.to_square
                            for move in self.board.legal_moves
                            if move.from_square
                            == chess.parse_square(self.current_start)
                        ]
                        if len(possible_moves) == 0:
                            print(
                                f"No possible moves from {self.current_start}")
                            self.current_start = None
                        self.render_board(self.current_start)
            self.clock.tick(60)

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


if __name__ == "__main__":

    def randomBot(moves: List[str], legalMoves: List[str]) -> str:
        """A simple random bot that selects a random legal move.

        Args:
            moves (List[str]): A list of the performed actions so far in UCI.
            getLegalMoves (List[str]): available moves in UCI format.

        Returns:
            str: A random legal move in UCI format.
        """
        return random.choice(legalMoves) if legalMoves else None

    # Example: start from a position after 1.e4 e5 2.Nf3 Nc6 3.Bb5
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    applet = ChessPlayerApplet(fen=test_fen, botActionFunction=randomBot)
    applet.run()
