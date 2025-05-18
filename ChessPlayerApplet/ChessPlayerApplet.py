import pygame
from pygame.locals import *
import chess
import chess.svg
import cairosvg
import io

pygame.init()
boardSize = 800
padding_x = 14/350 * boardSize
padding_y = 14/350 * boardSize
squareSize_x = (boardSize - 2*padding_x) / 8
squareSize_y = (boardSize - 2*padding_y) / 8
screen = pygame.display.set_mode((boardSize, boardSize))
clock = pygame.time.Clock()

board = chess.Board()


def pos2UCI(pos):
    # Convert pixel position to UCI format
    x, y = pos
    x = int((x-padding_x) / squareSize_x)
    y = int((y-padding_y) / squareSize_y)
    uci = chess.square_name(x + (7 - y) * 8)
    return uci


def render_board(start=None):
    # render board using chess library
    if start is not None:
        # Highlight the selected square and all possible moves from that square
        from_square = chess.parse_square(start)
        # Find all legal moves from the selected square
        possible_moves = [
            move.to_square for move in board.legal_moves if move.from_square == from_square]
        # Build fill dict: selected square in blue, possible moves in green
        fill = {from_square: "#1E90FF"}
        for sq in possible_moves:
            fill[sq] = "#32CD3280"  # 80 is ~50% alpha in hex
        boardSvg = chess.svg.board(
            board,
            size=boardSize,
            fill=fill
        )
    else:
        boardSvg = chess.svg.board(
            board,
            size=boardSize
        )

    # convert svg to png
    boardPng = cairosvg.svg2png(bytestring=boardSvg)

    # Make png the background
    background = pygame.image.load(io.BytesIO(boardPng)).convert_alpha()
    screen.blit(background, (0, 0))

    # Draw the board
    pygame.display.flip()


def uci2pos(uci):
    # Convert UCI format to pixel position
    square = chess.parse_square(uci)
    x = (square % 8) * squareSize_x + padding_x
    y = (7 - (square // 8)) * squareSize_y + padding_y
    return (x, y)


def main():
    currentStart = None
    render_board()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                currentPointer = pos2UCI(event.pos)
                if currentStart and currentPointer != currentStart:
                    move = chess.Move.from_uci(
                        f"{currentStart}{currentPointer}") if currentStart else None
                    print(f'Trying to move {currentStart} to {currentPointer}')
                    if move and move in board.legal_moves:
                        board.push(move)
                    currentStart = None
                    render_board(currentStart)
                else:
                    currentStart = currentPointer
                    render_board(currentStart)
                # can access properties with
                # proper notation(ex: event.y)
        clock.tick(60)


# Execute game:
main()
