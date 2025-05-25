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

Since a natural language should have a Zipf's exponent close to one (as per Shakespear) we tried estimating the chess exponent. From this analysis emerges that chess language has an exponent close to the zipfian case but from the plot we can see that it is pretty far from the ideal case, a line with negative slope.

We then tried to do the same analysis but for increasing move number and from the plot we can see a region where the Zipf's exponent is pretty much the same as the one of a natural language.

In addition, we wanted to investigate if the move language is more structured than english, so if moves appear frequently near the same moves. 

To do that we performed an n-gram analysis and we found that the frequency of n-grams in chess is much higher with respect to english, and that most of the top frequent n-grams are the openings of chess.




We decided to try fine-tuning GPT-2 to see whether this would improve its ability to suggest the next move given the history of a chess game. The model was fine-tuned using the move history of 90,000 games.

We then compared the standard GPT-2 and the fine-tuned version using zero-shot and few-shot prompting. We noticed that some parts of the prompt significantly improved the responses, while others were completely ignored.

Overall, the standard GPT-2 doesn't initially understand UCI moves, but starts to grasp them with few-shot examples—although it never suggests valid new moves. Surprisingly, the fine-tuned model performed slightly worse with few-shot prompting.

We evaluated the model on 10,000 sequences. The results weren't impressive, but sufficient to build a basic chess bot—very weak, capable of moving pieces most of the time, but with no strategic understanding.




