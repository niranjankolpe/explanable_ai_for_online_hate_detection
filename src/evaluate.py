import os
import json
import yaml
import pickle

import pandas as pd
import joblib
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report

from model_lstm import LSTMClassifier
from dataset_lstm import pad_sequence, preprocess
from predict_bert import load_bert_model, predict_bert_proba

# Configuration
MODEL_PATH      = "models/baseline/baseline_model.pkl"
VECTORIZER_PATH = "models/baseline/tfidf_vectorizer.pkl"
TEST_PATH       = "data/testset-levela.tsv"
LABELS_PATH     = "data/labels-levela.csv"

LSTM_MODEL_PATH = "models/lstm/lstm_model.pt"
LSTM_VOCAB_PATH = "models/lstm/lstm_vocab.pkl"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params   = params["lstm"]
MAX_LEN       = lstm_params["max_len"]
EMBEDDING_DIM = lstm_params["embedding_dim"]
HIDDEN_DIM    = lstm_params["hidden_dim"]
NUM_LAYERS    = lstm_params["num_layers"]
DROPOUT       = lstm_params["dropout"]

bert_params   = params["bert"]
BERT_MAX_LEN  = bert_params["max_len"]


def load_test_data():
    X_test = pd.read_csv(TEST_PATH, sep="\t")
    y_test = pd.read_csv(LABELS_PATH, header=None, names=["id", "label"])
    df     = X_test.merge(y_test, on="id")
    return df


def evaluate_lstm(df):
    print("\nEvaluating LSTM model...")

    with open(LSTM_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = LSTMClassifier(
        vocab_size=max(vocab.values()) + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(
        torch.load(LSTM_MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()

    texts  = df["tweet"].apply(preprocess)
    y_true = df["label"].map(lambda x: 1 if x == "OFF" else 0)

    sequences = []
    for text in texts:
        tokens = text.split()
        seq    = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
        seq    = pad_sequence(seq, MAX_LEN)
        sequences.append(seq)

    inputs = torch.tensor(sequences)
    with torch.no_grad():
        outputs = model(inputs)
        preds   = torch.argmax(outputs, dim=1).numpy()

    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, average="weighted")
    print(classification_report(y_true, preds, target_names=["NOT", "OFF"]))
    return acc, f1


def evaluate_bert(df):
    print("\nEvaluating DistilBERT model...")

    bert_model, tokenizer = load_bert_model()

    texts  = df["tweet"].apply(preprocess).tolist()
    y_true = [1 if l == "OFF" else 0 for l in df["label"]]

    # Run in batches to avoid memory issues
    batch_size = bert_params["batch_size"]
    all_preds  = []

    print(f"Starting for loop. Length of texts: {len(texts)}. Batch Size: {batch_size}")
    for i in range(0, len(texts), batch_size):
        print(f"Iteration {i} started...")
        batch_texts = texts[i:i + batch_size]
        #print("Got Batch texts")
        probs       = predict_bert_proba(batch_texts, bert_model, tokenizer, BERT_MAX_LEN)
        #print("Got probs")
        preds       = probs.argmax(axis=1).tolist()
        #print("Got preds")
        all_preds.extend(preds)
        #print("Extended preds")

    acc = accuracy_score(y_true, all_preds)
    f1  = f1_score(y_true, all_preds, average="weighted")
    print(classification_report(y_true, all_preds, target_names=["NOT", "OFF"]))
    return acc, f1


def main():
    print("Loading baseline model and vectorizer...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading test data...")
    df = load_test_data()

    X      = df["tweet"].str.lower()
    y_true = df["label"]

    print("Vectorizing...")
    X_tfidf = vectorizer.transform(X)

    print("Predicting...")
    preds = model.predict(X_tfidf)

    baseline_acc = accuracy_score(y_true, preds)
    baseline_f1  = f1_score(y_true, preds, average="weighted")
    print("\nBaseline (TF-IDF + LR):")
    print(classification_report(y_true, preds))

    lstm_acc, lstm_f1 = evaluate_lstm(df)
    bert_acc, bert_f1 = evaluate_bert(df)

    print("\n===== Final Comparison =====")
    print(f"Baseline  | Acc: {baseline_acc:.4f} | F1: {baseline_f1:.4f}")
    print(f"BiLSTM    | Acc: {lstm_acc:.4f} | F1: {lstm_f1:.4f}")
    print(f"DistilBERT| Acc: {bert_acc:.4f} | F1: {bert_f1:.4f}")

    metrics = {
        "baseline": {
            "accuracy":    float(baseline_acc),
            "f1_weighted": float(baseline_f1)
        },
        "lstm": {
            "accuracy":    float(lstm_acc),
            "f1_weighted": float(lstm_f1)
        },
        "bert": {
            "accuracy":    float(bert_acc),
            "f1_weighted": float(bert_f1)
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()