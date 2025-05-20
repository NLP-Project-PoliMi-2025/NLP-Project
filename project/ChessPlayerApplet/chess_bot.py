from typing import List, Dict
from project.models.lit_modules import SeqAnnotator
import torch
import numpy as np


class ChessBot:
    def __init__(
        self,
        weight_location: str,
        vocab_table: Dict[str, int],
        bot_starts: bool = False,
        epsilon: float = 0,  # 0 for greedy, 1 for random
    ):
        self.weight_location = weight_location
        self.moves_vocab_table = vocab_table["Moves"]
        self.bot_starts = bot_starts
        self.epsilon = epsilon  # epsilon greedy action selection

        if self.bot_starts:
            self.objective = vocab_table["result_seqs"]["1-0"]
        else:
            self.objective = vocab_table["result_seqs"]["0-1"]

        self.model = self.load_model(weight_location)

    def load_model(self, weights: str) -> SeqAnnotator:
        # Load the model from the given weights
        # This is a placeholder for the actual model loading logic
        model = SeqAnnotator.load_from_checkpoint(weights, map_location="cpu")
        model.eval()
        return model

    def __call__(self, past_moves: List[str], legal_moves: List[str]) -> str:
        # transform the moves to indices
        if legal_moves == []:
            raise ValueError("No legal moves available")
        past_moves_embeds = torch.tensor(
            [self.moves_vocab_table[move] for move in past_moves]
        )
        legal_moves_embeds = torch.tensor(
            [self.moves_vocab_table[move] for move in legal_moves]
        )

        # get the model prediction
        with torch.no_grad():
            _, hidden = self.model.forward(past_moves_embeds)
            legal_embeddings = self.model.embedding(legal_moves_embeds)
            logits, _ = self.model.rnn.forward(legal_embeddings, hidden)
            preds = self.model.fc_out.forward(logits)

        preds = preds[:, self.objective - 1]  # account for 0-indexing
        preds = torch.softmax(preds, dim=0)

        # Epsilon greedy action selection
        if torch.rand(1).item() < self.epsilon:
            action_idx = np.random.choice(len(legal_moves))
            rand_action = True
        else:
            action_idx = torch.argmax(preds).item()
            rand_action = False

        print(
            f"Predicted action index: {action_idx}, Certainty: {preds[action_idx].item()}, Entropy: {-torch.sum(preds * torch.log(preds + 1e-10)).item()}, Random: {rand_action}")
        action = legal_moves[action_idx]
        return action
