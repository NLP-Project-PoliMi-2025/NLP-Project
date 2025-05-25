# SCRIPT FOR THE VIDEO RECORDING

Since with this project we are looking into a **chess dataset**, the first question that came to our mind was:

    "Is the language of chess comparable to a natural language?"

To answer this question we performed some analysis on the dataset.

Firstly, since the dataset is generated using Stockfish, we analyzed how the data is distributed.
We found that the games are very long and probably the version of Stockfish used is not the best one since the tie percentage is not that high.




Performing an average distribution of the pieces moved, we can see that the  king is the most moved piece and this suggests the presence of long ending phases in games.

So we performed the same analysis looking at the lenght of the games, and from this heatmap we can truly see the opening phase (where pawns are moved) and the ending phase(where mostly king is moved).




To compare the chess move language with a natural language (english in this case) we decided to load another dataset with the plays of Shakespear.

We decided to apply the Zipf's Law to both dataset, that correlates frequencies of tokens with their rank.

It has been recorded that most natural languages have a Zipf's exponent that ranges between 0.7 and 1.4.

We computed the exponent for Shakespear and we saw that it aligns with the theoretical values for a natural language.
We estimated the exponent for chess. From this analysis emerges that chess language has an exponent close to the zipfian case but from the plot we can see that it is pretty far from the ideal case, a line with negative slope.

We then tried to do the same analysis for all the moves performed up to a certain move count, and from the plot we can see a region where the Zipf's exponent is in the range of natural languages.

In addition, we wanted to investigate if the move language is more structured than english, or whether there are moves that frequently appear together. 

To do that we performed an n-gram analysis and we found that the frequency of n-grams in chess is much higher with respect to english, and that most of the top frequent n-grams are the openings of chess.

##  EMBEDDINGS

We than asked ourselves whether thecontext informations can be collected at the token level.
So we trained Word2Vec both for Shakespeare and chess and we figured out that trying to retrieve a meaning for the move tokens was less powerful than to do it on english words, but we have still some  results such as clusters representing similar moves like promotions.


## GPT-2

We decided to try fine-tuning GPT-2 to see whether this would improve its ability to suggest the next move given the history of a chess game. The model was fine-tuned using the move history of 90,000 games.

We then compared the standard GPT-2 and the fine-tuned version using zero-shot and few-shot prompting. We noticed that some parts of the prompt significantly improved the responses, while others were completely ignored.

Overall, the standard GPT-2 doesn't initially understand UCI moves, but starts to grasp them with few-shot examples—although it never suggests valid new moves. Surprisingly, the fine-tuned model performed slightly worse with few-shot prompting.

We evaluated the model on 10,000 sequences. The results weren't impressive, but sufficient to build a basic chess bot—very weak, capable of moving pieces most of the time, but with no strategic understanding.

## PREDICTOR


## CHESSBOT