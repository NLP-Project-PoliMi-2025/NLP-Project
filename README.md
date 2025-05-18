# NLP_Project

Project repo for NLP course with Professor Mark Carman at Politecnico di Milano


# Moves

- Smith format
- start field, end field [optional: if piece captured]

    

# Termination

- CHECKMATE
- STALEMATE: one side has NO legal moves to make. If the king is NOT in check, but no piece can be moved without putting the king in check, then the game will end with a stalemate draw.
- SEVENTYFIVE_MOVES: If seventy-five moves are made without a pawn move or capture being made, the game is drawn unless the seventy-fifth move delivers a checkmate
- INSUFFICIENT_MATERIAL: The insufficient mating material rule says that the game is immediately declared a draw if there is no way to end the game in checkmate
- FIVEFOLD_POSITION: If the same position occurs five times, then the game is immediately terminated as a draw


# Result:

- one side wins -> 1-0 or 0-1
- draw -> 1/2 - 1/2



## Training Game Annotation

python -m project train-game-annotation --label <dataframe column> --model-type <model-type>    
