import chess
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, SVG
from chess import svg
from cairosvg import svg2png


class MovePlotter:
    def __init__(self, title: str):
        self.reset()
        self.title = title
        self.moves = []
        self.board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")  # Empty board

    def moveToIndex(self, move: str):
        return ord(move[0]) - ord('a'), 8 - int(move[1])

    def addMove(self, move: str):
        fromSquare = chess.parse_square(move[:2])
        toSquare = chess.parse_square(move[2:4])
        self.moves.append((fromSquare, toSquare))

    def plot(self, justBoard: bool = False):
        # Plot the "from-to" distributions
        from_squares = [move[0] for move in self.moves]
        to_squares = [move[1] for move in self.moves]

        from_ranks, from_files = zip(
            *[divmod(square, 8) for square in from_squares])
        to_ranks, to_files = zip(*[divmod(square, 8) for square in to_squares])

        # From-square distribution
        # Accumulate moves in an array
        from_square_counts = np.zeros((8, 8), dtype=int)
        to_square_counts = np.zeros((8, 8), dtype=int)

        for rank, file in zip(from_ranks, from_files):
            from_square_counts[rank, file] += 1

        for rank, file in zip(to_ranks, to_files):
            to_square_counts[rank, file] += 1

        if not justBoard:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(self.title)
            # From-square distribution
            axes[0].imshow(from_square_counts, cmap="Blues", origin="upper")
            axes[0].set_title("From-Square Distribution")
            axes[0].set_xlabel("File")
            axes[0].set_ylabel("Rank")
            axes[0].set_xticks(range(8))
            axes[0].set_xticklabels([chr(i + ord('a')) for i in range(8)])
            axes[0].set_yticks(range(8))
            axes[0].set_yticklabels([str(i + 1) for i in range(8)])
            # flip y-axis to match chessboard coordinates
            axes[0].invert_yaxis()
            axes[0].set_aspect('equal')

            # To-square distribution
            axes[1].imshow(to_square_counts, cmap="Reds", origin="upper")
            axes[1].set_title("To-Square Distribution")
            axes[1].set_xlabel("File")
            axes[1].set_ylabel("Rank")
            axes[1].set_xticks(range(8))
            axes[1].set_xticklabels([chr(i + ord('a')) for i in range(8)])
            axes[1].set_yticks(range(8))
            axes[1].set_yticklabels([str(i + 1) for i in range(8)])
            # flip y-axis to match chessboard coordinates
            axes[1].invert_yaxis()
            axes[1].set_aspect('equal')

            plt.tight_layout()
            plt.show()

        # Plot the chessboard with arrows
        arrows = [chess.svg.Arrow(from_sq, to_sq, color="#0000cc55")
                  for from_sq, to_sq in self.moves]
        svg_board = chess.svg.board(
            self.board,
            arrows=arrows,
            size=350
        )
        display(SVG(svg_board))

    def getBoardWidget(self, size: int = 200):
        # Plot the chessboard with arrows
        arrows = [chess.svg.Arrow(from_sq, to_sq, color="#0000cc55")
                  for from_sq, to_sq in self.moves]
        svg_board = chess.svg.board(
            self.board,
            arrows=arrows,
            size=size
        )

        png_board = svg2png(bytestring=svg_board)
        board_widget = widgets.Image(
            value=png_board,
            format='png',
            width=350,
            height=350
        )
        return board_widget

    def reset(self):
        self.moves = []
        # Reset to empty board
        self.board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
