from abc import ABC, abstractmethod
import matplotlib
from typing import List, Dict
from nlpChess.models.lit_modules import SeqAnnotator
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

sns.set_theme()
matplotlib.use("TkAgg")  # Add this at the top, after imports


class _ChessBot(ABC):
    @abstractmethod
    def load_model(self, weights: str) -> object:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, past_moves: List[str], legal_moves: List[str]) -> str:
        raise NotImplementedError


class LSTMChessBot(_ChessBot):
    def __init__(
        self,
        weight_location: str,
        vocab_table: Dict[str, Dict[str, int]],
        bot_starts: bool = False,
        epsilon: float = 0,  # 0 for greedy, 1 for random
    ):
        self.weight_location = weight_location
        self.moves_vocab_table: Dict[str, int] = vocab_table["Moves"]
        self.outcome_vocab_table: Dict[str, int] = vocab_table["result_seqs"]
        self.bot_starts = bot_starts
        self.epsilon = epsilon  # epsilon greedy action selection
        # Order the outcome_vocab_table keys by their value
        ordered_outcome_keys = [
            k
            for k, v in sorted(
                self.outcome_vocab_table.items(), key=lambda item: item[1]
            )
        ]
        self.performanceTracker = ChessBotPerformanceTrackerHandler(
            {
                "Number of possible Moves": [0],
                "Certainty": [0],
                "Entropy": [0],
                "Outcome Prediction": [[]],
            },
            {
                "Outcome Prediction": ordered_outcome_keys,
            },
        )

        if self.bot_starts:
            self.objective = self.outcome_vocab_table["1-0"]
        else:
            self.objective = self.outcome_vocab_table["0-1"]

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

            current_evaluation, _ = self.model.forward(past_moves_embeds)

        current_evaluation = current_evaluation.numpy()

        preds = preds[:, self.objective - 1]  # account for 0-indexing
        preds = torch.softmax(preds, dim=0)

        # Epsilon greedy action selection
        if torch.rand(1).item() < self.epsilon:
            action_idx = np.random.choice(len(legal_moves))
            rand_action = True
        else:
            action_idx = torch.argmax(preds).item()
            rand_action = False

        certainty = preds[action_idx].item()
        entropy = -torch.sum(preds * torch.log(preds + 1e-10)).item()
        print(
            f"Predicted action index: {action_idx}, Certainty: {certainty}, Entropy: {entropy}, Random: {rand_action}"
        )
        self.performanceTracker.pushUpdate(
            ChessBotPerformanceTrackerUpdate(
                {
                    "Number of possible Moves": len(legal_moves),
                    "Certainty": certainty,
                    "Entropy": entropy,
                    "Outcome Prediction": current_evaluation,
                }
            )
        )
        action = legal_moves[action_idx]
        return action


class ChessBotPerformanceTrackerUpdate:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics


class ChessBotPerformanceTrackerHandler:
    def __init__(self, metrics: Dict, labels: Dict, updatePeriod: float = 0.1):
        self.queue = mp.Queue()
        self.plotterProcess = mp.Process(
            target=ChessBotPerformanceTrackerHandler.plotterProcess_function,
            args=(self.queue, metrics, labels, updatePeriod),
        )
        self.plotterProcess.start()

    def plotterProcess_function(
        queue: mp.Queue, metrics: Dict, labels: Dict, updatePeriod: float
    ):
        print("[Plotter process started]")
        import seaborn as sns

        sns.set_theme()
        import time

        plotter = ChessBotPerformanceTracker(metrics=metrics, labels=labels)

        while True:
            while not queue.empty():
                update: ChessBotPerformanceTrackerUpdate = queue.get()
                plotter.processUpdate(update)

            # plt.pause(updatePeriod)
            plotter.fig.canvas.draw_idle()
            plotter.fig.canvas.flush_events()
            time.sleep(updatePeriod)

    def pushUpdate(self, update: ChessBotPerformanceTrackerUpdate):
        self.queue.put(update)


class ChessBotPerformanceTracker:
    """This class can store and dynamically plot the performance of the chess bot.
    It can be used to track the performance of the bot over time and visualize the results.
    Now, each metric is plotted in its own subplot.
    """

    def __init__(self, metrics: Dict, labels: Dict):
        self.metrics = metrics
        self.metricsNames = list(metrics.keys())
        self.num_metrics = len(metrics)
        self.labels = labels
        plt.ion()
        self.fig, self.axes = plt.subplots(
            self.num_metrics, 1, figsize=(5, 2.5 * self.num_metrics), squeeze=False
        )
        self.fig.canvas.manager.set_window_title(
            "Chess Bot Performance Tracker")
        self.fig.suptitle("Chess Bot Performance")
        plt.show()
        self.update_plot()

    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics to the tracker and update the plot.
        Args:
            metrics (Dict[str, float]): A dictionary of metrics to add.
        """
        for metric in metrics:
            if metric in self.metricsNames:
                self.metrics[metric].append(metrics[metric])
            else:
                raise ValueError(
                    f"Metric {metric} not in metricsNames: {self.metricsNames}"
                )

        self.update_plot()

    def override_metrics(self, metrics: Dict[str, List]):
        """Override the metrics with new values.
        Args:
            metrics (Dict[str, List[float]]): A dictionary of metrics to override.
        """
        for metric in metrics:
            if metric in self.metricsNames:
                self.metrics[metric] = metrics[metric]
            else:
                self.metrics[metric] = []

        self.update_plot()

    def getLabel(self, metric: str):
        """Get the label for a given metric.
        Args:
            metric (str): The metric to get the label for.
        Returns:
            str: The label for the metric.
        """
        if metric in self.labels:
            return (
                self.labels[metric]
                if self.labels[metric] or len(self.labels[metric]) > 0
                else None
            )
        else:
            return None

    def update_plot(self):
        for idx, metric in enumerate(self.metricsNames):
            ax = self.axes[idx, 0]
            ax.clear()
            ax.set_title(metric)
            ax.set_xlabel("Games Played")
            ax.set_ylabel(metric)
            if (
                isinstance(self.metrics[metric], list)
                and self.metrics[metric]
                and isinstance(self.metrics[metric][0], list)
            ):
                for i, vals in enumerate(self.metrics[metric]):
                    ax.plot(vals, marker="o", label=self.labels[metric][i])
            else:
                ax.plot(
                    range(0, len(self.metrics[metric]) * 2, 2),
                    self.metrics[metric],
                    label=self.getLabel(metric),
                    marker="o",
                )
            ax.legend()
        plt.tight_layout()

    def processUpdate(self, update: ChessBotPerformanceTrackerUpdate):
        newMetrics = update.metrics.copy()
        for metric in newMetrics:
            if metric in self.metrics:
                import collections.abc

                if isinstance(
                    newMetrics[metric], collections.abc.Iterable
                ) and not isinstance(newMetrics[metric], (str, bytes)):
                    self.metrics[metric] = newMetrics[metric]
                else:
                    self.metrics[metric].append(newMetrics[metric])
            else:
                self.metrics[metric] = [newMetrics[metric]]
        self.update_plot()
