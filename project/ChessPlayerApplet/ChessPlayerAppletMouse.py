import pygame
from pygame.locals import *
import chess
from typing import List
import random
from project.ChessPlayerApplet.ChessPlayerApplet import ChessPlayerApplet


class ChessPlayerAppletMouse(ChessPlayerApplet):
    def __init__(self, board_size=800, fen=None, botActionFunction=None):
        """ChessPlayerApplet is a class that creates a chess applet using Pygame and the python-chess library.
        It allows the user to play chess against a bot or another player. The applet displays the chessboard and pieces,
        handles user input, and updates the board state.

        Args:
            board_size (int, optional): The pixel size of the board. Defaults to 800.
            fen (str, optional): The start board configuration, if None the defult is picked. Defaults to None.
            botActionFucntion (function: (List[str], List[str]) -> str, optional): function that takes a list of the performed actions so far in UCI and a list of available moves to get the available moves. If None the user input are taken as bot actions. Defaults to None.
        """
        super().__init__(board_size, fen, botActionFunction)

    def pos2uci(self, pos):
        """Convert pixel position to UCI format.
        Args:
            pos (tuple): The pixel position (x, y) on the board.
        Returns:
            str: The UCI format of the square (e.g., "e2").
        """
        # Convert pixel position to UCI format
        x, y = pos
        x = int((x - self.padding_x) / self.square_size_x)
        y = int((y - self.padding_y) / self.square_size_y)
        uci = chess.square_name(x + (7 - y) * 8)
        return uci

    def uci2pos(self, uci):
        """Convert UCI format to pixel position.
        Args:
            uci (str): The UCI format of the square (e.g., "e2").
        Returns:
            tuple: The pixel position (x, y) on the board.
        """
        # Convert UCI format to pixel position
        square = chess.parse_square(uci)
        x = (square % 8) * self.square_size_x + self.padding_x
        y = (7 - (square // 8)) * self.square_size_y + self.padding_y
        return (x, y)

    def run(self):
        """Main loop of the applet. Handles user input and updates the board state."""
        self.render_board()
        game_over = False  # Track if the game is over
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                if game_over:
                    continue  # Ignore input if game is over
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

                            if self.board.is_game_over():
                                print("Game over:", self.board.result())
                                game_over = True
                                self.render_board(self.current_start)
                                continue

                            if self.botActionFunction is not None:
                                legal_moves = self.getLegalMoves()
                                legal_moves = [move.uci() for move in legal_moves]
                                self.performAction(
                                    chess.Move.from_uci(
                                        self.botActionFunction(
                                            self.UCImoves, legal_moves
                                        )
                                    )
                                )

                                if self.board.is_game_over():
                                    print("Game over:", self.board.result())
                                    game_over = True
                                    self.render_board()
                                    continue

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
                            print(f"No possible moves from {self.current_start}")
                            self.current_start = None
                        self.render_board(self.current_start)
            self.clock.tick(60)


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
    applet = ChessPlayerAppletMouse(
        fen=test_fen,
        botActionFunction=randomBot,
    )
    applet.run()
