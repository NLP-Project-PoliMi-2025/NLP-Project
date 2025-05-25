from huggingface_hub import PyTorchModelHubMixin

from nlpChess.models.BiLSTM import BiLSTM_NER
from nlpChess.models.lit_modules import SeqAnnotator


class SeqAnnotatorHFWrapper(
    SeqAnnotator,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/ruhrpott/LSTM-chess-result-2-512",
    pipeline_tag="text-classification",
    license="mit",
):
    def __init__(
        self,
        n_target_classes,
        label,
        vocab_size,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=0.001,
        model_type="lstm",
        ignore_index=None,
        word2vec=None,
        freeze_embeddings=False,
        label_counts=None,
        bidirectional=False,
        logging_last_token_metrics=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_target_classes,
            label,
            vocab_size,
            d_model,
            n_layers,
            n_heads,
            dropout,
            lr,
            model_type,
            ignore_index,
            word2vec,
            freeze_embeddings,
            label_counts,
            bidirectional,
            logging_last_token_metrics,
            *args,
            **kwargs,
        )
        
class SimpleLSTMHFWrapper(BiLSTM_NER,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/ruhrpott/LSTM-chess-piece-1-128",
    pipeline_tag="text-classification",
    license="mit",
):
    def __init__(self, embedding_dim, hidden_dim, num_outcomes, n_layers = 1):
        super().__init__(embedding_dim, hidden_dim, num_outcomes, n_layers)
        
        



if __name__ == "__main__":
    model = SeqAnnotatorHFWrapper.load_from_checkpoint("lstm-best-v3.ckpt")
    model.save_pretrained("LSTM-chess-result-2-512")

    # push to the hub
    model.push_to_hub("ruhrpott/LSTM-chess-result-2-512", token=None)
    # reload
    model = SeqAnnotatorHFWrapper.from_pretrained("ruhrpott/LSTM-chess-result-2-512")