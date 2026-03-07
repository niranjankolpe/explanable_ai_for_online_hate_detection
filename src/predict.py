import sys
import torch
import pickle

from .model_lstm import LSTMClassifier
from .dataset_lstm import pad_sequence

import yaml

MODEL_PATH = "models/lstm/lstm_model.pt"
VOCAB_PATH = "models/lstm/lstm_vocab.pkl"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MAX_LEN = params["lstm"]["max_len"]


def load_model():

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    return model, vocab


def preprocess(text):
    return text.lower().strip()


def predict(text, model, vocab):
    text = preprocess(text)
    tokens = text.split()

    seq = [vocab.get(t, 1) for t in tokens]

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
        seq = [vocab.get(t, 1) for t in tokens]
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
