"""
tests/test_core.py
Unit tests for preprocessing, dataset utilities, and vocabulary.

Run: python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import preprocess_common, preprocess_lstm
from dataset    import Vocabulary, pad_sequence


# ── preprocess_common ─────────────────────────────────────────────────────────
assert preprocess_common("Hello @user http://link.com!!!") == "hello"
assert preprocess_common("  YOU ARE IDIOT  ") == "you are idiot"
assert preprocess_common("") == ""
assert preprocess_common("@mention http://url.com") == ""

# ── preprocess_lstm ───────────────────────────────────────────────────────────
assert preprocess_lstm("Hello @user http://link.com!!!") == "hello"
assert preprocess_lstm("  YOU ARE IDIOT  ") == "you are idiot"
assert preprocess_lstm("") == ""

# ── pad_sequence ──────────────────────────────────────────────────────────────
assert pad_sequence([1, 2, 3], 5)       == [1, 2, 3, 0, 0]
assert pad_sequence([1, 2, 3, 4, 5, 6], 4) == [1, 2, 3, 4]
assert pad_sequence([], 3)              == [0, 0, 0]

# ── Vocabulary ────────────────────────────────────────────────────────────────
import pandas as pd
vocab = Vocabulary(max_size=10)
vocab.build_vocab(pd.Series(["hello world", "hello python"]))
assert "<PAD>" in vocab.word2idx
assert "<UNK>" in vocab.word2idx
assert "hello" in vocab.word2idx
assert vocab.word2idx["<PAD>"] == 0
assert vocab.word2idx["<UNK>"] == 1

print("All tests passed.")
