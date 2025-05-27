All the main analysis has been performed in Jupyter notebooks, which are located in the `notebooks` directory.
The available notebooks are:
- [`bert_retrieval`](./bert_retrieval.ipynb): our attempt at using plain BERT embeddings to do games retrieval. The biggest finding is that adding an "embedding prompt" before the move sequence significantly improves the results.

- [`data_exploration`](./data_exploration.ipynb): this notebook contains the initial data exploration and analysis, including the distribution of moves, game lengths, and other statistics.

- [`final`](./final.ipynb): this notebook contains a sum up of the analysis and results of the project, including the comparison of chess move sequences with English text, and the application of Zipf's Law. It also includes the zero and few shot evaluation of gpt2 on chess move sequences.

- [`FullGamesEmbeddings`](./FullGamesEmbeddings.ipynb): this notebook contains an in depth analysis of the embeddings of out pre trained lstm models. We trained the same models on different tast to see if the produced embeddings could be used for other down stream tasks.

- [`game_entropy](./game_entropy.ipynb): this notebook contains the analysis of the entropy of chess games, comparing the distribution of moves and how it changes over the course of a game. This can be used to distinguish openings, mid-game and endings.

- [`game_tree_analysis`](./game_tree_analysis.ipynb): this notebook contains the analysis of the game tree of chess games, including the number of moves, branching factor, and other statistics. It also includes the analysis of the game termination types and results.

- [`gpt2ChessBot`](./gpt2ChessBot.ipynb): contains the code for the utilization of the fine tuned gpt2 model to generate chess moves with a playable interface.

- [`Llama`](./Llama.ipynb): in here you can find our attempt at fine tuning a Llama model using an Alpaca-like method. This approach didn't turn out to be very effective, but it is still worth a look.

- [`lstm_analysis`](./lstm_analysis.ipynb): this notebook contains the analysys of the performances of various LSTM models that we trained on different tasks, including check and move prediction. The findings about the distribution of the crossentropy loss of the models trhough a game are consistent with what we found in the game entropy notebook.

- [`OutcomeClassification`](./OutcomeClassification.ipynb): this is the main notebook to train our prediction models on different tasks.

- [`pad_token_percentage`](./pad_token_percentage.ipynb): this notebook analysis the deminishing return of bigger batch sizes in training due to the need of inserting padding tokens in the sequences. Even with just a batch size of 32 we can expect to use 50% of the tokens as padding, which is a significant waste of resources.

- [`Word2VecEmbeddings`](./Word2VecEmbeddings.ipynb): this notebook contains the analysis of the Word2Vec embeddings of chess moves, including the visualization of the embeddings and the comparison with other embeddings. Here you can play with an interactive applet to see the "synonyms" of a chess move.

- [`ZipfLaw`](./ZipfLaw.ipynb): this notebook contains the analysis of Zipf's Law applied to chess move sequences, comparing the distribution of moves with English text and other languages. The main findings is that although chess behaves quite differently from natural language, it still follows a Zipf-like distribution, with a few moves being very common and many moves being rare, expecially in the midgame.
