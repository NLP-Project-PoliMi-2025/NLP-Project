import pygame
from pygame.locals import *
import chess
import chess.svg
import cairosvg
import io


class ChessPlayerApplet:
    def __init__(self, board_size=800, fen=None):
        pygame.init()
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
                            self.board.push(move)
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


if __name__ == "__main__":
    # Example: start from a position after 1.e4 e5 2.Nf3 Nc6 3.Bb5
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    applet = ChessPlayerApplet(fen=test_fen)
    applet.run()
