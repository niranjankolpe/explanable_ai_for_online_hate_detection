import os
import sys
import json
import yaml
import pickle

import pandas as pd
import joblib
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow

from model_lstm import LSTMClassifier
from dataset_lstm import pad_sequence, preprocess
from predict_bert import load_bert_model, predict_bert_proba

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params = params["lstm"]
bert_params = params["bert"]
BERT_MAX_LEN = bert_params["max_len"]

EMBEDDING_DIM = lstm_params["embedding_dim"]
HIDDEN_DIM    = lstm_params["hidden_dim"]
NUM_LAYERS    = lstm_params["num_layers"]
DROPOUT       = lstm_params["dropout"]
MAX_LEN       = lstm_params["max_len"]


def load_test_data(subtask):
    subtask_config = params["subtasks"][subtask]
    X_test = pd.read_csv(subtask_config["test_file"], sep="\t")
    y_test = pd.read_csv(subtask_config["labels_file"], header=None, names=["id", "label"])
    df     = X_test.merge(y_test, on="id")
    df     = df[df["label"].notna()].copy()
    return df


def evaluate_baseline(df, subtask):
    print(f"\nEvaluating Baseline for Subtask {subtask.upper()}...")
    model      = joblib.load(f"models/baseline_{subtask}/baseline_model.pkl")
    vectorizer = joblib.load(f"models/baseline_{subtask}/tfidf_vectorizer.pkl")

    X      = df["tweet"].str.lower()
    y_true = df["label"]
    preds  = model.predict(vectorizer.transform(X))

    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, average="weighted")
    print(classification_report(y_true, preds))
    return acc, f1


def evaluate_lstm(df, subtask):
    print(f"\nEvaluating BiLSTM for Subtask {subtask.upper()}...")

    labels    = params["subtasks"][subtask]["labels"]
    label2idx = {label: idx for idx, label in enumerate(labels)}

    with open(f"models/lstm_{subtask}/lstm_vocab.pkl", "rb") as f:
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
        torch.load(f"models/lstm_{subtask}/lstm_model.pt", map_location=torch.device("cpu"))
    )
    model.eval()

    texts  = df["tweet"].apply(preprocess)
    y_true = df["label"].map(label2idx)

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
    print(classification_report(y_true, preds, target_names=labels))
    return acc, f1


def evaluate_bert(df, subtask):
    print(f"\nEvaluating DistilBERT for Subtask {subtask.upper()}...")

    labels    = params["subtasks"][subtask]["labels"]
    label2idx = {label: idx for idx, label in enumerate(labels)}

    bert_model, tokenizer = load_bert_model(subtask)

    texts  = df["tweet"].apply(preprocess).tolist()
    y_true = df["label"].map(label2idx).tolist()

    batch_size = bert_params["batch_size"]
    all_preds  = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        probs       = predict_bert_proba(batch_texts, bert_model, tokenizer, BERT_MAX_LEN)
        preds       = probs.argmax(axis=1).tolist()
        all_preds.extend(preds)

    acc = accuracy_score(y_true, all_preds)
    f1  = f1_score(y_true, all_preds, average="weighted")
    print(classification_report(y_true, all_preds, target_names=labels))
    return acc, f1


def evaluate_subtask(subtask):
    print(f"\n{'='*40}")
    print(f"Evaluating Subtask {subtask.upper()}")
    print(f"{'='*40}")

    df = load_test_data(subtask)

    baseline_acc, baseline_f1 = evaluate_baseline(df, subtask)
    lstm_acc, lstm_f1         = evaluate_lstm(df, subtask)
    bert_acc, bert_f1         = evaluate_bert(df, subtask)

    print(f"\n===== Subtask {subtask.upper()} Results =====")
    print(f"Baseline   | Acc: {baseline_acc:.4f} | F1: {baseline_f1:.4f}")
    print(f"BiLSTM     | Acc: {lstm_acc:.4f} | F1: {lstm_f1:.4f}")
    print(f"DistilBERT | Acc: {bert_acc:.4f} | F1: {bert_f1:.4f}")

    return {
        "baseline": {"accuracy": float(baseline_acc), "f1_weighted": float(baseline_f1)},
        "lstm":     {"accuracy": float(lstm_acc),     "f1_weighted": float(lstm_f1)},
        "bert":     {"accuracy": float(bert_acc),     "f1_weighted": float(bert_f1)}
    }


def main():
    subtask = sys.argv[1] if len(sys.argv) > 1 else None

    subtasks_to_run = [subtask] if subtask in ["a", "b", "c"] else ["a", "b", "c"]

    all_metrics = {}
    for st in subtasks_to_run:
        all_metrics[f"subtask_{st}"] = evaluate_subtask(st)

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hate_detection_evaluation")
    with mlflow.start_run():
        for st, metrics in all_metrics.items():
            for model, vals in metrics.items():
                mlflow.log_metric(f"{st}_{model}_accuracy",    vals["accuracy"])
                mlflow.log_metric(f"{st}_{model}_f1_weighted", vals["f1_weighted"])

    print("\nMetrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()