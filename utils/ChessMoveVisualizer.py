import chess
import chess.svg
from IPython.display import display, SVG
import ipywidgets as widgets
from cairosvg import svg2png


class ChessMoveVisualizer:
    def __init__(self, model, board_size=350, n_similar_moves=10):
        self.model = model
        self.board = chess.Board()
        self.board_size = board_size
        self.n_similar_moves = n_similar_moves

        # Initialize board display widget
        self.board_svg = chess.svg.board(
            board=self.board, size=self.board_size)
        self.board_png = svg2png(bytestring=self.board_svg)
        self.board_display_widget = widgets.Image(
            format='png', value=self.board_png, width=self.board_size, height=self.board_size)
        self.board_display_widget.layout.width = f"{self.board_size}px"
        self.board_display_widget.layout.height = f"{self.board_size}px"

        # Create widgets
        self.move_input = widgets.Text(
            description="Move:", placeholder="e.g., e2e4")
        self.make_move_button = widgets.Button(description="Make Move")
        self.output = widgets.Output()

        # Attach event handlers
        self.move_input.observe(self.on_move, names="value")
        self.make_move_button.on_click(self.on_make_move_button_click)

        # Display initial board and widgets
        display(self.board_display_widget, self.move_input,
                self.make_move_button, self.output)

    def display_board(self):
        self.board_svg = chess.svg.board(
            board=self.board, size=self.board_size)
        self.board_png = svg2png(bytestring=self.board_svg)
        self.board_display_widget.value = self.board_png

    def display_move(self, move):
        from_square = chess.parse_square(move[:2])
        to_square = chess.parse_square(move[2:4])
        self.board_svg = chess.svg.board(
            board=self.board, size=self.board_size, arrows=[(from_square, to_square)])
        self.board_png = svg2png(bytestring=self.board_svg)
        self.board_display_widget.value = self.board_png

    def find_similar_move(self, move):
        if move in self.model.wv:
            similar_moves = self.model.wv.most_similar(
                move, topn=self.n_similar_moves)
            return similar_moves
        else:
            return "Move not in vocabulary"

    def display_most_similar_moves(self, move, most_similar_moves):
        self.output.clear_output()
        with self.output:
            if isinstance(most_similar_moves, str):
                print(most_similar_moves)
            else:
                print(f"Most similar moves:")
                for move_, similarity in most_similar_moves:
                    print(f"\t{move_}: {similarity:.2f}")

        from_square = chess.parse_square(move[:2])
        to_square = chess.parse_square(move[2:4])
        move_arrow = chess.svg.Arrow(
            from_square, to_square, color="rgba(0, 255, 0, 0.5)")

        arrows = []
        for similar_move, similarity in most_similar_moves:
            from_square = chess.parse_square(similar_move[:2])
            to_square = chess.parse_square(similar_move[2:4])
            opacity = max(0.1, min(1.0, similarity))
            arrows.append(chess.svg.Arrow(from_square, to_square,
                          color=f"rgba(0, 0, 255, {opacity})"))

        arrows.append(move_arrow)

        self.board_svg = chess.svg.board(
            board=self.board, size=self.board_size, arrows=arrows)
        self.board_png = svg2png(bytestring=self.board_svg)
        self.board_display_widget.value = self.board_png

    def on_move(self, change):
        move = self.move_input.value
        try:
            if move in self.model.wv:
                similar_move = self.find_similar_move(move)
                self.display_most_similar_moves(move, similar_move)
        except ValueError:
            self.output.clear_output()
            with self.output:
                print("Invalid move. Please try again.")

    def on_make_move_button_click(self, button):
        move = self.move_input.value
        try:
            self.board.push_uci(move)

            if move in self.model.wv:
                similar_move = self.find_similar_move(move)
                self.display_most_similar_moves(move, similar_move)
        except ValueError:
            self.output.clear_output()
            with self.output:
                print("Invalid move. Please try again.")
        self.display_board()
