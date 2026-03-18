import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report

import os
import json

import torch
import pickle

from model_lstm import LSTMClassifier
from dataset_lstm import pad_sequence

# Configuration
MODEL_PATH      = "models/baseline/baseline_model.pkl"
VECTORIZER_PATH = "models/baseline/tfidf_vectorizer.pkl"
TEST_PATH       = "data/testset-levela.tsv"
LABELS_PATH     = "data/labels-levela.csv"

LSTM_MODEL_PATH = "models/lstm/lstm_model.pt"
LSTM_VOCAB_PATH = "models/lstm/lstm_vocab.pkl"
MAX_LEN = 50


# Load Test Data
def load_test_data():
    X_test = pd.read_csv(TEST_PATH, sep="\t")
    y_test = pd.read_csv(LABELS_PATH, header=None, names=["id", "label"])

    # Merge labels with test set
    df = X_test.merge(y_test, on="id")
    return df

def evaluate_lstm(df):

    print("\nEvaluating LSTM model...")

    with open(LSTM_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(
    vocab_size=max(vocab.values()) + 1,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.5)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    model.eval()

    X = df["tweet"].str.lower()
    y_true = df["label"].map(lambda x: 1 if x == "OFF" else 0)

    sequences = list()

    for text in X:
        tokens = text.split()
        seq = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
        seq = pad_sequence(seq, MAX_LEN)
        sequences.append(seq)

    inputs = torch.tensor(sequences)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")

    return acc, f1

# Main Evaluation
def main():

    print("Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading test data...")
    df = load_test_data()

    X = df["tweet"].str.lower()
    y_true = df["label"]

    print("Vectorizing...")
    X_tfidf = vectorizer.transform(X)

    print("Predicting...")
    preds = model.predict(X_tfidf)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")

    print("\nEvaluating baseline model...")
    baseline_acc = acc
    baseline_f1 = f1

    lstm_acc, lstm_f1 = evaluate_lstm(df)

    metrics = {
        "baseline": {
            "accuracy": float(baseline_acc),
            "f1_weighted": float(baseline_f1)
        },
        "lstm": {
            "accuracy": float(lstm_acc),
            "f1_weighted": float(lstm_f1)
        }
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved metrics:")
    print(metrics)

# Driver code
if __name__ == "__main__":
    main()
