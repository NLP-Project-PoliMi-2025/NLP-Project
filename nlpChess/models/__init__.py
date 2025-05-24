from gensim.models.word2vec import Word2Vec

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
Word2VecChess = Word2Vec.load(os.path.join(current_dir, "word2vec100.model"))
Word2VecShakespear = Word2Vec.load(os.path.join(
    current_dir, "word2vec_shakespear.model"))
