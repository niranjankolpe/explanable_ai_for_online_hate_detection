import sys
import re
import yaml
import pickle

import torch

try:
    from .model_lstm import LSTMClassifier
    from .dataset_lstm import pad_sequence, preprocess
except ImportError:
    from model_lstm import LSTMClassifier
    from dataset_lstm import pad_sequence, preprocess

MODEL_PATH = "models/lstm/lstm_model.pt"
VOCAB_PATH = "models/lstm/lstm_vocab.pkl"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params   = params["lstm"]
MAX_LEN       = lstm_params["max_len"]
EMBEDDING_DIM = lstm_params["embedding_dim"]
HIDDEN_DIM    = lstm_params["hidden_dim"]
NUM_LAYERS    = lstm_params["num_layers"]
DROPOUT       = lstm_params["dropout"]


def load_model():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(
        vocab_size=max(vocab.values()) + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    return model, vocab


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
    sequences = []
    for text in texts:
        text = preprocess(text)          # same preprocess() as predict()
        if not text:
            seq = [vocab["<UNK>"]]
        else:
            tokens = text.split()
            seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
        seq = pad_sequence(seq, MAX_LEN)  # module-level MAX_LEN, no yaml reload
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
    print(f"Input: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {round(confidence, 4)}")


if __name__ == "__main__":
    main()