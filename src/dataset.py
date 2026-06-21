"""
dataset.py
Vocabulary builder and PyTorch Dataset for BiLSTM training.
"""

import torch
from torch.utils.data import Dataset
from collections import Counter

from preprocess import preprocess_common


def pad_sequence(seq: list, max_len: int) -> list:
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


class Vocabulary:
    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, sentences) -> None:
        counter = Counter()
        for sentence in sentences:
            counter.update(preprocess_common(sentence).split())
        for idx, (word, _) in enumerate(counter.most_common(self.max_size - 2), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    def numericalize(self, text: str) -> list:
        return [
            self.word2idx.get(token, self.word2idx["<UNK>"])
            for token in text.split()
        ]


class OLIDDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int = 25):
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        text    = preprocess_common(self.texts.iloc[idx])
        seq     = self.vocab.numericalize(text)
        padded  = pad_sequence(seq, self.max_len)
        label   = int(self.labels.iloc[idx])
        return torch.tensor(padded), torch.tensor(label)
