import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from project.chess_utils import get_board_states


def connect_chess_db(db_name: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(db_name, check_same_thread=False)
    cursor = conn.cursor()
    return conn, cursor


def create_tables(cursor: sqlite3.Cursor):
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY,
        result TEXT,
        termination TEXT
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS moves (
        id INTEGER PRIMARY KEY,
        game_id INTEGER,
        move_number INTEGER,
        move_id INTEGER,
        board_fen_id INTEGER,
        FOREIGN KEY(game_id) REFERENCES games(id),
        FOREIGN KEY(move_id) REFERENCES move_collection(id),
        FOREIGN KEY(game_id) REFERENCES games(id)
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS board_states (
        id INTEGER PRIMARY KEY,
        board_fen TEXT,
        UNIQUE(board_fen)
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS move_collection (
        id INTEGER PRIMARY KEY,
        move TEXT,
        UNIQUE(move)
    )
    """
    )


def insert_if_not_exists(cursor, table, column, value):
    cursor.execute(f"INSERT OR IGNORE INTO {table} ({column}) VALUES (?)", (value,))
    cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
    return cursor.fetchone()[0]


def insert_game(
    cursor: sqlite3.Cursor,
    move_sequence: List[str],
    board_fens: List[str],
    result: str,
    termination: str,
):
    """
    Inserts a game and its moves into the database.

    - move_sequence: list of SAN move strings
    - board_fens: list of FEN strings *after each move*
    - result: string (e.g. '1-0')
    - termination: string (e.g. 'checkmate')
    """

    # 1. Insert into games
    cursor.execute(
        """
        INSERT INTO games (result, termination) VALUES (?, ?)
    """,
        (result, termination),
    )
    game_id = cursor.lastrowid

    # 2. Insert moves and board states
    for move_number, (move_str, fen_str) in enumerate(
        zip(move_sequence, board_fens), start=1
    ):
        # Insert (or get) move id
        move_id = insert_if_not_exists(cursor, "move_collection", "move", move_str)

        # Insert (or get) board_fen_id
        board_fen_id = insert_if_not_exists(
            cursor, "board_states", "board_fen", fen_str
        )

        # Insert into moves
        cursor.execute(
            """
            INSERT INTO moves (game_id, move_number, move_id, board_fen_id)
            VALUES (?, ?, ?, ?)
        """,
            (game_id, move_number, move_id, board_fen_id),
        )

    return game_id


def create_chess_db(name: str, parquet_files: List[str], subsample: int = None):
    conn, cursor = connect_chess_db(name)
    create_tables(cursor)

    for idx, parquet_file in enumerate(parquet_files):
        ds = pd.read_parquet(parquet_file)
        if subsample is not None:
            ds_index = np.random.choice(len(ds), subsample)
            ds = ds.iloc[ds_index]
        for game_idx in tqdm(
            np.arange(len(ds)),
            desc=f"Load games into DB ({idx + 1}/{len(parquet_files)})",
        ):
            moves, termination, result = ds.iloc[game_idx]
            board_fen = get_board_states(moves)
            insert_game(cursor, moves, board_fen, result, termination)
    conn.commit()
    conn.close()


# =============================================== FETCH DATA ===============================================
def fetch_games_with_moves(
    filters: List[Tuple[str, str, str]] = None,
    columns: List[str] = None,
    connection: sqlite3.Connection = None,
    db_path: str = "chess.db",
) -> pd.DataFrame:
    new_connection = False
    if connection is None:
        connection = sqlite3.connect(db_path)
        new_connection = True

    colmap = {
        "game_id": "g.id AS game_id",
        "result": "g.result",
        "termination": "g.termination",
        "move_number": "m.move_number",
        "move": "mc.move",
        "board_fen": "bs.board_fen",
    }
    selected = [colmap[c] for c in (columns or colmap.keys())]

    where_clause = ""
    params = []
    if filters:
        conditions = []
        for col, op, val in filters:
            qualified_col = {
                "move": "mc.move",
                "board_fen": "bs.board_fen",
                "move_number": "m.move_number",
                "game_id": "g.id",
                "result": "g.result",
                "termination": "g.termination",
            }.get(col, f"g.{col}")
            conditions.append(f"{qualified_col} {op} ?")
            params.append(val)
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
    SELECT {', '.join(selected)}
    FROM games g
    JOIN moves m ON g.id = m.game_id
    JOIN move_collection mc ON m.move_id = mc.id
    JOIN board_states bs ON m.board_fen_id = bs.id
    {where_clause}
    ORDER BY g.id, m.move_number
    """

    df = pd.read_sql_query(query, connection, params=params)
    if new_connection:
        connection.close()
    return df


def fetch_games(
    filters: List[Tuple[str, str, str]] = None,
    columns: List[str] = None,
    connection: sqlite3.Connection = None,
    db_path: str = "chess.db",
) -> pd.DataFrame:
    new_connection = False
    if connection is None:
        connection = sqlite3.connect(db_path)
        new_connection = True

    # Default columns
    colmap = {
        "id": "g.id AS game_id",
        "result": "g.result",
        "termination": "g.termination",
    }
    selected = [colmap[c] for c in (columns or colmap.keys())]

    where_clause = ""
    params = []
    if filters:
        conditions = [f"g.{col} {op} ?" for col, op, val in filters]
        params = [val for _, _, val in filters]
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
    SELECT {', '.join(selected)}
    FROM games g
    {where_clause}
    ORDER BY g.id
    """

    df = pd.read_sql_query(query, connection, params=params)
    if new_connection:
        connection.close()
    return df


def fetch_moves(
    filters: List[Tuple[str, str, str]] = None,
    columns: List[str] = None,
    connection: sqlite3.Connection = None,
    db_path: str = "chess.db",
) -> pd.DataFrame:
    new_connection = False
    if connection is None:
        connection = sqlite3.connect(db_path)
        new_connection = True
    connection = sqlite3.connect(db_path)

    colmap = {
        "move_id": "m.id AS move_id",
        "game_id": "m.game_id",
        "move_number": "m.move_number",
        "move": "mc.move",
        "board_fen": "bs.board_fen",
    }
    selected = [colmap[c] for c in (columns or colmap.keys())]

    where_clause = ""
    params = []
    if filters:
        conditions = []
        for col, op, val in filters:
            qualified_col = {
                "move": "mc.move",
                "board_fen": "bs.board_fen",
                "move_number": "m.move_number",
                "game_id": "m.game_id",
            }.get(col, f"m.{col}")
            conditions.append(f"{qualified_col} {op} ?")
            params.append(val)
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
    SELECT {', '.join(selected)}
    FROM moves m
    JOIN move_collection mc ON m.move_id = mc.id
    JOIN board_states bs ON m.board_fen_id = bs.id
    {where_clause}
    ORDER BY m.game_id, m.move_number
    """

    df = pd.read_sql_query(query, connection, params=params)
    if new_connection:
        connection.close()
    return df
