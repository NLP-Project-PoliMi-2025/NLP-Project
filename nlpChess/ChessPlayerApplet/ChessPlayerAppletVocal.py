import pygame
import chess
from pygame.locals import *
from nlpChess.ChessPlayerApplet.ChessPlayerApplet import ChessPlayerApplet
from nlpChess.utils.audioPlayer import AudioPlayer
import multiprocess as mp
import time


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
        self.toAudioPlayer, toApplet = mp.Pipe()

        print("Starting audio player process")
        self.audio_player_process = mp.Process(
            target=ChessPlayerAppletVocal.audioPlayerProcess,
            args=(toApplet,),
        )
        self.audio_player_process.start()

    def audioPlayerProcess(toApplet: mp.Pipe):
        """Process that handles the audio player.
        This process is used to handle the audio player in a separate process.
        """
        print("[Audio player process started]")
        audioPlayer = AudioPlayer()

        audioPlayer.read_text(
            "Welcome to the chess game. Press a key and say the move you want to make."
        )

        while True:
            if toApplet.poll():
                op = toApplet.recv()
                if op == "speak":
                    content = toApplet.recv()
                    if "Game over" in content:
                        audioPlayer.read_text(content)
                        return
                    audioPlayer.read_text(content)
                elif op == "get":
                    moveToPerform = audioPlayer.get_move()
                    toApplet.send(moveToPerform)
            time.sleep(0.1)

    def readText(self, text):
        self.toAudioPlayer.send("speak")
        self.toAudioPlayer.send(text)

    def listenMove(self):
        """Listens for a move from the audio player.
        This method is used to listen for a move from the audio player.
        """
        self.toAudioPlayer.send("get")
        move = self.toAudioPlayer.recv()
        return move

    def run(self):
        """Main loop of the applet. Handles user input and updates the board state."""
        self.render_board()
        game_over = False  # Track if the game is over
        PlayerPlayed = False

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
                    move = self.listenMove()
                    while move not in self.board.legal_moves:
                        error_text = "Illegal move. Please try again."
                        self.readText(error_text)
                        print(error_text)
                        move = self.listenMove()

                    self.performAction(move)
                    self.current_start = move.from_square

                    if self.board.is_game_over():
                        end_text = f"Game over: {self.board.result()}"
                        print(end_text)
                        self.readText(end_text)
                        self.render_board(self.current_start)
                        game_over = True
                    elif self.botActionFunction is not None:
                        legal_moves = self.getLegalMoves()
                        legal_moves = [move.uci() for move in legal_moves]
                        self.performAction(
                            chess.Move.from_uci(
                                self.botActionFunction(
                                    self.UCImoves, legal_moves)
                            )
                        )
                        text = f"Bot played {self.UCImoves[-1]}, "

                        if self.board.is_game_over():
                            text += f"Game over: {self.board.result()}"
                            print(text)
                            self.readText(text)
                            game_over = True
                            self.render_board()
                            continue
                        else:
                            text += "your turn."

                        print(text)
                        self.readText(text)

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
