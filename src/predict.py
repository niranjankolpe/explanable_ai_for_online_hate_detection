import sys
import yaml
import pickle

import torch

try:
    from .model_lstm import LSTMClassifier
    from .dataset_lstm import pad_sequence, preprocess
except ImportError:
    from model_lstm import LSTMClassifier
    from dataset_lstm import pad_sequence, preprocess

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params   = params["lstm"]
MAX_LEN       = lstm_params["max_len"]
EMBEDDING_DIM = lstm_params["embedding_dim"]
HIDDEN_DIM    = lstm_params["hidden_dim"]
NUM_LAYERS    = lstm_params["num_layers"]
DROPOUT       = lstm_params["dropout"]


def load_model(subtask="a"):
    vocab_path = f"models/lstm_{subtask}/lstm_vocab.pkl"
    model_path = f"models/lstm_{subtask}/lstm_model.pt"
    labels     = params["subtasks"][subtask]["labels"]

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(
        vocab_size=max(vocab.values()) + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=len(labels)
    )
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model, vocab


def predict(text, model, vocab, subtask="a"):
    labels = params["subtasks"][subtask]["labels"]
    text   = preprocess(text)
    tokens = text.split()
    seq    = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
    seq    = pad_sequence(seq, MAX_LEN)
    inputs = torch.tensor([seq])

    with torch.no_grad():
        outputs    = model(inputs)
        probs      = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = labels[pred.item()]
    return label, confidence.item()


def predict_proba(texts, model, vocab):
    sequences = []
    for text in texts:
        text = preprocess(text)
        if not text:
            seq = [vocab["<UNK>"]]
        else:
            tokens = text.split()
            seq    = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
        seq = pad_sequence(seq, MAX_LEN)
        sequences.append(seq)

    inputs = torch.tensor(sequences)
    with torch.no_grad():
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1).numpy()
    return probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"text\" [subtask]")
        return
    text    = sys.argv[1]
    subtask = sys.argv[2] if len(sys.argv) > 2 else "a"
    model, vocab = load_model(subtask)
    label, confidence = predict(text, model, vocab, subtask)
    print(f"Input: {text}")
    print(f"Subtask {subtask.upper()} Prediction: {label}")
    print(f"Confidence: {round(confidence, 4)}")


if __name__ == "__main__":
    main()