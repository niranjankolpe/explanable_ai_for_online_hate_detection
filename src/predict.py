import sys
import torch
import pickle

from .model_lstm import LSTMClassifier
from .dataset_lstm import pad_sequence

import yaml

import re

MODEL_PATH = "models/lstm/lstm_model.pt"
VOCAB_PATH = "models/lstm/lstm_vocab.pkl"


import os

# print("MODEL PATH:", MODEL_PATH)
# print("EXISTS:", os.path.exists(MODEL_PATH))

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MAX_LEN = params["lstm"]["max_len"]


def load_model():
    # print("Inside load_model()")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    # print("Vocabulary loaded into vocab")

    model = LSTMClassifier(
    vocab_size=max(vocab.values()) + 1,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    # print("LSTM Model Loaded with CPU")

    model.eval()
    # print("Ran model.eval()")
    return model, vocab


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def predict(text, model, vocab):
    # print("Inside predict()")

    text = preprocess(text)
    tokens = text.split()
    # print("Text preprocessed and split")

    seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
    # print("Sequence list generated")

    seq = pad_sequence(seq, MAX_LEN)
    # print("Padded sequence list")

    inputs = torch.tensor([seq])
    # print("Got inputs for tensor")

    with torch.no_grad():
        # print("Inside with control")

        outputs = model(inputs)
        # print("Generated outputs")

        probs = torch.softmax(outputs, dim=1)
        # print(" Generated probs")

        confidence, pred = torch.max(probs, dim=1)
        # print("\Generated confidence and predn\n")

    label = "OFF" if pred.item() == 1 else "NOT"
    # print("Done OFF and NOT labelling")

    return label, confidence.item()

def predict_proba(texts, model, vocab):
    # print("Inside predict_proba()")

    # load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # print("Loaded params.yaml file")

    MAX_LEN = params["lstm"]["max_len"]
    # print("Extracted MAX_LEN")

    sequences = []

    # print("Starting for loop:")
    # print(f"Original Texts: {texts}")
    # print("Number of texts:", len(texts))
    # print("Empty texts:", sum(1 for t in texts if not t.strip()))
    for text in texts:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            seq = [vocab["<UNK>"]]
        else:
            tokens = text.lower().split()
            seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]

        seq = pad_sequence(seq, MAX_LEN)
        sequences.append(seq)
    inputs = torch.tensor(sequences)
    # print(f"Inputs: {inputs}")

    with torch.no_grad():
        # print("Inside with")

        outputs = model(inputs)
        # print("Outputs generated")

        probs = torch.softmax(outputs, dim=1).numpy()
        # print("probs generated")
    return probs

def main():

    if len(sys.argv) < 2:
        # print("Usage: python predict.py \"text to classify\"")
        return

    text = sys.argv[1]

    model, vocab = load_model()

    label, confidence = predict(text, model, vocab)

    # # print("\nInput:", text)
    # # print("Prediction:", label)
    # # print("Confidence:", round(confidence, 4))


if __name__ == "__main__":
    main()
