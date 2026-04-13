import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset_lstm import preprocess, pad_sequence, Vocabulary

# Test preprocess()
assert preprocess("Hello @user http://link.com!!!") == "hello"
assert preprocess("  YOU ARE IDIOT  ") == "you are idiot"
assert preprocess("") == ""

# Test pad_sequence()
assert pad_sequence([1, 2, 3], 5) == [1, 2, 3, 0, 0]
assert pad_sequence([1, 2, 3, 4, 5, 6], 4) == [1, 2, 3, 4]
assert pad_sequence([], 3) == [0, 0, 0]

# Test Vocabulary
vocab = Vocabulary(max_size=10)
vocab.build_vocab(["hello world", "hello python"])
assert "<PAD>" in vocab.word2idx
assert "<UNK>" in vocab.word2idx
assert "hello" in vocab.word2idx
assert vocab.word2idx["<PAD>"] == 0
assert vocab.word2idx["<UNK>"] == 1

print("All tests passed.")