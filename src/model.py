"""
model.py
Bidirectional LSTM classifier.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int = 128,
        hidden_dim:    int = 128,
        num_layers:    int = 2,
        dropout:       float = 0.5,
        num_classes:   int = 2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        embedded              = self.embedding(x)
        _, (hidden, _)        = self.lstm(embedded)
        hidden_cat            = self.dropout(
            torch.cat((hidden[-2], hidden[-1]), dim=1)
        )
        return self.fc(hidden_cat)
