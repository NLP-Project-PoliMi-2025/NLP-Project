import pygame
import chess
from pygame.locals import *
from project.ChessPlayerApplet.ChessPlayerApplet import ChessPlayerApplet
from project.utils.audioPlayer import AudioPlayer


class ChessPlayerAppletVocal(ChessPlayerApplet):
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
        self.audio_player = AudioPlayer()


    def run(self):
        """Main loop of the applet. Handles user input and updates the board state.
        """
        self.audio_player.read_text("Welcome to the chess game. Press a key and say the move you want to make.")
        self.render_board()
        game_over = False  # Track if the game is over
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                if game_over:
                    continue  # Ignore input if game is over
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        return
                    while True:
                        move = self.audio_player.get_move()
                        text = f"Trying to move {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
                        print(text)
                        self.audio_player.read_text(text)
                        if move in self.board.legal_moves:
                            self.performAction(move)
                            self.current_start = move.from_square
                            if self.board.is_game_over():
                                end_text = f"Game over: {self.board.result()}"
                                print(end_text)
                                self.audio_player.read_text(end_text)
                                self.render_board(self.current_start)
                                break
                            break
                        else:
                            error_text = "Illegal move. Please try again."
                            print(error_text)
                            self.audio_player.read_text(error_text)

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
                        text = f"Bot played {self.UCImoves[-1]}, your turn."
                        print(text)
                        self.audio_player.read_text(text)

                        if self.board.is_game_over():
                            text = f"Game over: {self.board.result()}"
                            print(text)
                            self.audio_player.read_text(text)
                            game_over = True
                            self.render_board()
                            continue

                        self.current_start = None
                        self.render_board(self.current_start)
                    else:
                        self.current_start = move.from_square
                        # Check if there are possible moves from the current square
                        possible_moves = [
                            move.to_square
                            for move in self.board.legal_moves
                            if move.from_square
                            == chess.parse_square(self.current_start)
                        ]
                        if len(possible_moves) == 0:
                            text = f"No possible moves from {self.current_start}"
                            print(text)
                            self.audio_player.read_text(text)
                            self.current_start = None
                        self.render_board(self.current_start)
            self.clock.tick(60)

    # def handle_player_move(self):
    #     """Handles the player's move by getting the move from the audio player and performing it on the board.
    #     This method is used to handle the player's move in the game.
    #     Returns:
    #         bool: True if the game is over, False otherwise.
    #     """
    #     while True:
    #         move = self.audio_player.get_move()
    #         text = f"Trying to move {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
    #         print(text)
    #         self.audio_player.read_text(text)
    #         if move in self.board.legal_moves:
    #             self.performAction(move)
    #             self.current_start = move.from_square
    #             if self.board.is_game_over():
    #                 end_text = f"Game over: {self.board.result()}"
    #                 print(end_text)
    #                 self.audio_player.read_text(end_text)
    #                 self.render_board(self.current_start)
    #                 return True
    #             return False
    #         else:
    #             error_text = "Illegal move. Please try again."
    #             print(error_text)
    #             self.audio_player.read_text(error_text)

    # def handle_bot_move(self):
    #     """Handles the bot's move by getting the move from the bot action function and performing it on the board.
    #     This method is used to handle the bot's move in the game.
    #     Returns:
    #         bool: True if the game is over, False otherwise.
    #     """
    #     if self.botActionFunction is not None:
    #         legal_moves = [move.uci() for move in self.getLegalMoves()]
    #         bot_move_uci = self.botActionFunction(self.UCImoves, legal_moves)
    #         bot_move = chess.Move.from_uci(bot_move_uci)
    #         self.performAction(bot_move)
    #         self.current_start = None
    #         if self.board.is_game_over():
    #             end_text = f"Game over: {self.board.result()}"
    #             print(end_text)
    #             self.audio_player.read_text(end_text)
    #             self.render_board()
    #             return True
    #         bot_text = f"Bot played {self.UCImoves[-1]}, your turn."
    #         print(bot_text)
    #         self.audio_player.read_text(bot_text)
    #     return False

    # def run(self):
    #     """Main loop of the applet. Handles user input and updates the board state.
    #     This method is used to run the main loop of the applet.
    #     """
    #     self.render_board()
    #     game_over = False
    #     self.audio_player.read_text("Welcome to the chess game. Say the move you want to make.")
    #     # Initial player move
    #     game_over = self.handle_player_move()
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == QUIT:
    #                 pygame.quit()
    #                 return
    #             if game_over:
    #                 continue

    #             # Bot move (if applicable)
    #             if self.botActionFunction is not None:
    #                 game_over = self.handle_bot_move()
    #                 if game_over:
    #                     continue

    #                 # Player move after bot
    #                 self.handle_player_move()
    #                 if game_over:
    #                     continue

    #                 self.current_start = None
    #                 self.render_board(self.current_start)
    #             else:
    #                 # Handle player move
    #                 game_over = self.handle_player_move()
    #                 if game_over:
    #                     continue

    #                 self.current_start = None
    #                 self.render_board(self.current_start)
    #         self.clock.tick(60)