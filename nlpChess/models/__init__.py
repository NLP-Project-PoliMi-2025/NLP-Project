from gensim.models.word2vec import Word2Vec
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .BiLSTM_NER import BiLSTM_NER
import yaml

import os
from enum import Enum

current_dir = os.path.dirname(os.path.abspath(__file__))
Word2VecChess = Word2Vec.load(os.path.join(current_dir, "word2vec.model"))
Word2VecShakespear = Word2Vec.load(os.path.join(
    current_dir, "word2vec_shakespear.model"))


with open(os.path.join(current_dir, "./move_map.yaml"), "r") as f:
    MOVE_MAP = yaml.safe_load(f)
    # remove None values
    MOVE_MAP = {v: k for k, v in MOVE_MAP.items() if v is not None}


class PretrainedModels(Enum):
    def __init__(self, modelPath: str, nOutcomes: int = 128):
        self.modelPath = modelPath
        self.nOutcomes = nOutcomes

    def getArgumentsFromPath(path: str) -> list:
        pathList = path.split("-")[::-1]
        return [int(x) for x in pathList[:2] if x.isdigit()]

    def loadModel(self):
        stateDictPath = hf_hub_download(
            repo_id=self.modelPath, filename="model.safetensors"
        )
        state_dict = load_file(stateDictPath)
        model = BiLSTM_NER(
            100,
            self.nOutcomes,
            *PretrainedModels.getArgumentsFromPath(self.modelPath)
        )
        model.load_state_dict(state_dict)
        return model

    NEXT_TOKEN = "ruhrpott/LSTM-chess-next-move-2-128", 1959
    CAPTURES = "ruhrpott/LSTM-chess-captures-1-128", 7
    CHECKS = "ruhrpott/LSTM-chess-checks-1-128", 2
    PIECES = "ruhrpott/LSTM-chess-piece-1-128", 6
