import sys
import torch
import pickle

from .model_lstm import LSTMClassifier
from .dataset_lstm import pad_sequence

import yaml

import re

MODEL_PATH = "models/lstm/lstm_model.pt"
VOCAB_PATH = "models/lstm/lstm_vocab.pkl"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MAX_LEN = params["lstm"]["max_len"]


def load_model():

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(
    vocab_size=max(vocab.values()) + 1,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.5)
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    return model, vocab


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def predict(text, model, vocab):
    text = preprocess(text)
    tokens = text.split()

    seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]

    seq = pad_sequence(seq, MAX_LEN)

    inputs = torch.tensor([seq])

    with torch.no_grad():

        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    label = "OFF" if pred.item() == 1 else "NOT"

    return label, confidence.item()

def predict_proba(texts, model, vocab):

    # load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    MAX_LEN = params["lstm"]["max_len"]

    sequences = []

    for text in texts:
        tokens = text.lower().split()
        seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
        seq = pad_sequence(seq, MAX_LEN)
        sequences.append(seq)

    inputs = torch.tensor(sequences)

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).numpy()

    return probs

def main():

    if len(sys.argv) < 2:
        print("Usage: python predict.py \"text to classify\"")
        return

    text = sys.argv[1]

    model, vocab = load_model()

    label, confidence = predict(text, model, vocab)

    print("\nInput:", text)
    print("Prediction:", label)
    print("Confidence:", round(confidence, 4))


if __name__ == "__main__":
    main()
