from datasets import load_dataset, concatenate_datasets


def loadDataset():
    return load_dataset("PaoloGinefra03/ChessGamesForNlp")


def loadShakespeareDataset():
    return load_dataset("ruhrpott/Shakespeare")


def loadConcatenatedShakespeareDataset():
    datasets = loadShakespeareDataset()
    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]
    return concatenate_datasets([train, validation, test])


def loadConcatenatedDataset():
    datasets = loadDataset()
    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]
    return concatenate_datasets([train, validation, test])
