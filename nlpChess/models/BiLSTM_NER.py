import torch.nn as nn
import torch


class BiLSTM_NER(nn.Module):
    def __init__(self, embedding_dim, num_outcomes, hidden_dim, nlayers: int = 1):
        super(BiLSTM_NER, self).__init__()
        # Bidirectional LSTM; we set batch_first=True to have input like [batch, seq_len, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=False, batch_first=True, num_layers=nlayers)
        # Fully connected layer to map hidden state coming from LSTM to output labels
        # (the hidden state is a concatenation of two LSTM outputs since it is bidirectional)
        self.fc = nn.Linear(hidden_dim, num_outcomes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        # lstm_out: [batch_size, seq_len, hidden_dim*2]
        lstm_out, _ = self.lstm.forward(x)
        # logits: [batch_size, seq_len, num_tags]
        logits = self.fc(lstm_out)
        # probabs: [batch_size, seq_len, num_tags]
        probabs = torch.nn.functional.softmax(logits, dim=-1)
        return probabs, logits
