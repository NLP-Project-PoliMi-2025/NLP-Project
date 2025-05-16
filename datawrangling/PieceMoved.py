import sqlite3
from tqdm import tqdm

def fen_to_board(fen):
    """Convert FEN string to 8x8 board array."""
    board_fen = fen.split()[0]
    board = []
    for row in board_fen.split('/'):
        expanded_row = []
        for char in row:
            if char.isdigit():
                expanded_row.extend(['.'] * int(char))
            else:
                expanded_row.append(char)
        board.append(expanded_row)
    return board

def detect_move_and_capture(fen_before, fen_after):
    """Detect the piece moved and captured from FEN strings."""
    # Convert FEN strings to board arrays
    board_before = fen_to_board(fen_before)
    board_after = fen_to_board(fen_after)

    from_square = None
    to_square = None
    piece_moved = None
    piece_captured = None

    for rank in range(8):
        for file in range(8):
            b = board_before[rank][file]
            a = board_after[rank][file]
            if b != a:
                if b != '.' and a == '.':
                    # This is the from-square (piece disappeared)
                    from_square = (rank, file)
                    piece_moved = b
                elif a != '.' and (b == '.' or b.islower() != a.islower()):
                    # This is the to-square (piece appeared or captured)
                    to_square = (rank, file)
                    if b != '.':
                        piece_captured = b

    return piece_moved.lower() if piece_moved else None, \
           piece_captured.lower() if piece_captured else None


if(__name__ == "__main__"):
    # Connect to the SQLite database
    connection = sqlite3.connect('chess_games.db')
    cursor = connection.cursor()
    cursor.execute('''
        SELECT game_id, move_number, move, board_fen FROM moves
    ''')
    moves = cursor.fetchall()
    cursor.execute('''
        DROP TABLE IF EXISTS pieces
    ''')
    # Create a table to store the piece moved data
    cursor.execute('''
        CREATE TABLE pieces (
            game_id INTEGER,
            move_number INTEGER,
            piece TEXT,
            captured TEXT,
            PRIMARY KEY (game_id, move_number)
            FOREIGN KEY (game_id) REFERENCES games(game_id)
            FOREIGN KEY (move_number) REFERENCES moves(move_number)
        )   
    ''')
    # Process the moves and detect pieces moved and captured
    for i in tqdm(range(len(moves) - 1), desc="Processing moves"):
        game_id, move_number, move, fen_before = moves[i]
        _, next_move_number, _, fen_after = moves[i + 1]

        # Ensure we're comparing within the same game
        if game_id == moves[i + 1][0]:
            piece, piece_captured = detect_move_and_capture(fen_before, fen_after)
            # Insert the moved piece into the database
            cursor.execute('''
                INSERT INTO pieces (game_id, move_number, piece, captured)
                VALUES (?, ?, ?, ?)
            ''', (game_id, move_number, piece, piece_captured))
    connection.commit()
    cursor.close()
    connection.close()


