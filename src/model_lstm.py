import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        num_classes=2
        ):
        super(LSTMClassifier, self).__init__()

        # Word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        # x shape: (batch_size, seq_len)

        embedded = self.embedding(x)
        # shape: (batch_size, seq_len, embedding_dim)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        # hidden shape: (num_layers, batch_size, hidden_dim)

        last_hidden = hidden[-1]
        # shape: (batch_size, hidden_dim)

        dropped = self.dropout(last_hidden)

        logits = self.fc(dropped)
        # shape: (batch_size, num_classes)

        return logits