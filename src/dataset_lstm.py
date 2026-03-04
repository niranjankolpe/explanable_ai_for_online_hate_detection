import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter


class Vocabulary:
    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, sentences):
        counter = Counter()

        for sentence in sentences:
            tokens = sentence.split()
            counter.update(tokens)

        most_common = counter.most_common(self.max_size - 2)

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def numericalize(self, sentence):
        tokens = sentence.split()

        return [
            self.word2idx[token] if token in self.word2idx else self.word2idx["<UNK>"]
            for token in tokens
        ]


def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


class OLIDDataset(Dataset):
    def __init__(self, text_series, label_series, vocab, max_len=50):
        self.texts = text_series
        self.labels = label_series
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx].lower()

        sequence = self.vocab.numericalize(text)

        padded = pad_sequence(sequence, self.max_len)

        label = 1 if self.labels.iloc[idx] == "OFF" else 0

        return torch.tensor(padded), torch.tensor(label)