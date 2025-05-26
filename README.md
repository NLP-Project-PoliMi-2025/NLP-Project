# NLP Project: Chess Move Language Analysis & ChessBot

This repository contains the code and resources for the NLP course project at Politecnico di Milano, supervised by Professor Mark Carman. The project explores the intersection of chess and natural language processing, analyzing chess move sequences as a language and building models for annotation and move prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Chess Move Representation](#chess-move-representation)
- [Game Termination Types](#game-termination-types)
- [Game Results](#game-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Game Annotation](#training-game-annotation)
  - [Starting the Chess Bot](#starting-the-chess-bot)
- [Notebooks](#notebooks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

The main goal of this project is to investigate whether the language of chess moves exhibits properties similar to natural language. We analyze a large chess dataset, compare it with English text (e.g., Shakespeare), and apply NLP techniques such as Zipf's Law. Additionally, we develop models for annotating chess games and implement a ChessBot capable of move prediction.

---

## Dataset

- The chess dataset is generated using Stockfish and contains thousands of games.
- Each game is represented as a sequence of moves in UCI/Smith notation.
- Additional datasets (e.g., Shakespeare's plays) are used for comparative linguistic analysis.

---

## Chess Move Representation

- **Format:** UCI/Smith notation (e.g., e2e4, g1f3)
- **Fields:**  
  - Start field  
  - End field  
  - [Optional] Promotion piece

---

## Game Termination Types

- **CHECKMATE:** One side wins by checkmate.
- **STALEMATE:** No legal moves, king not in check.
- **SEVENTYFIVE_MOVES:** 75 moves without pawn move or capture (unless checkmate).
- **INSUFFICIENT_MATERIAL:** Not enough material to checkmate.
- **FIVEFOLD_POSITION:** Same position occurs five times.

---

## Game Results

- **White wins:** `1-0`
- **Black wins:** `0-1`
- **Draw:** `1/2-1/2`

---
## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/NLP-Project-PoliMi-2025.git
   cd NLP-Project-PoliMi-2025
   pip install -e .
    ```
2. For the fineTunedGpt2 download the model.safetensor from [`here`](https://drive.google.com/file/d/1kZR8brp_6XpRQ37lstUAJOvd6Fx-FgRs/view?usp=sharing) into  'ChessBOT_GPT2/Model90K/model'
## Project Structure

---

All the main analysis has been performed in Jupyter notebooks, which are located in the `notebooks` directory.
The available notebooks are:
- [`bert_retrieval`](./notebooks/bert_retrieval.ipynb): our attempt at using plain BERT embeddings to do games retrieval. The biggest finding is that adding an "embedding prompt" before the move sequence significantly improves the results.

- [`data_exploration`](./notebooks/data_exploration.ipynb): this notebook contains the initial data exploration and analysis, including the distribution of moves, game lengths, and other statistics.

- [`final`](./notebooks/final.ipynb): this notebook contains a sum up of the analysis and results of the project, including the comparison of chess move sequences with English text, and the application of Zipf's Law. It also includes the zero and few shot evaluation of gpt2 on chess move sequences.

- [`FullGamesEmbeddings`](./notebooks/FullGamesEmbeddings.ipynb): this notebook contains an in depth analysis of the embeddings of out pre trained lstm models. We trained the same models on different tast to see if the produced embeddings could be used for other down stream tasks.

- [`game_entropy](./notebooks/game_entropy.ipynb): this notebook contains the analysis of the entropy of chess games, comparing the distribution of moves and how it changes over the course of a game. This can be used to distinguish openings, mid-game and endings.

- [`game_tree_analysis`](./notebooks/game_tree_analysis.ipynb): this notebook contains the analysis of the game tree of chess games, including the number of moves, branching factor, and other statistics. It also includes the analysis of the game termination types and results.

- [`gpt2ChessBot`](./notebooks/gpt2ChessBot.ipynb): contains the code for the utilization of the fine tuned gpt2 model to generate chess moves with a playable interface.

- [`Llama`](./notebooks/Llama.ipynb): in here you can find our attempt at fine tuning a Llama model using an Alpaca-like method. This approach didn't turn out to be very effective, but it is still worth a look.

- [`lstm_analysis`](./notebooks/lstm_analysis.ipynb): this notebook contains the analysys of the performances of various LSTM models that we trained on different tasks, including check and move prediction. The findings about the distribution of the crossentropy loss of the models trhough a game are consistent with what we found in the game entropy notebook.

- [`OutcomeClassification`](./notebooks/OutcomeClassification.ipynb): this is the main notebook to train our prediction models on different tasks.

- [`pad_token_percentage`](./notebooks/pad_token_percentage.ipynb): this notebook analysis the deminishing return of bigger batch sizes in training due to the need of inserting padding tokens in the sequences. Even with just a batch size of 32 we can expect to use 50% of the tokens as padding, which is a significant waste of resources.

- [`Word2VecEmbeddings`](./notebooks/Word2VecEmbeddings.ipynb): this notebook contains the analysis of the Word2Vec embeddings of chess moves, including the visualization of the embeddings and the comparison with other embeddings. Here you can play with an interactive applet to see the "synonyms" of a chess move.

- [`ZipfLaw`](./notebooks/ZipfLaw.ipynb): this notebook contains the analysis of Zipf's Law applied to chess move sequences, comparing the distribution of moves with English text and other languages. The main findings is that although chess behaves quite differently from natural language, it still follows a Zipf-like distribution, with a few moves being very common and many moves being rare, expecially in the midgame.
