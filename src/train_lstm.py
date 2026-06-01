"""
train_lstm.py
BiLSTM training for subtask a/b/c.

Usage: python src/train_lstm.py [a|b|c]
"""

import os
import sys
import pickle
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.pytorch

from dataset import Vocabulary, OLIDDataset
from model   import BiLSTMClassifier

DATA_PATH    = "data/olid-training-v1.0.tsv"
RANDOM_STATE = 42

with open("params.yaml") as f:
    params = yaml.safe_load(f)

p = params["lstm"]


def train(subtask: str) -> None:
    cfg        = params["subtasks"][subtask]
    column     = cfg["column"]
    labels     = cfg["labels"]
    num_classes = len(labels)
    label2idx  = {lbl: i for i, lbl in enumerate(labels)}

    print(f"\nTraining BiLSTM — Subtask {subtask.upper()}")

    df          = pd.read_csv(DATA_PATH, sep="\t")
    df          = df[df[column].notna()].copy()
    texts       = df["tweet"].astype(str).tolist()
    label_list  = df[column].map(label2idx).tolist()    # FIX: assigned only once

    X_train, X_val, y_train, y_val = train_test_split(
        texts, label_list,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=label_list,
    )

    vocab = Vocabulary(max_size=p["vocab_size"])
    vocab.build_vocab(pd.Series(X_train))

    train_ds = OLIDDataset(pd.Series(X_train), pd.Series(y_train), vocab, p["max_len"])
    val_ds   = OLIDDataset(pd.Series(X_val),   pd.Series(y_val),   vocab, p["max_len"])
    train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=p["batch_size"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=p["embedding_dim"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
        num_classes=num_classes,
    ).to(device)

    counts  = Counter(y_train)
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"lstm_subtask_{subtask}")

    val_acc, val_f1, avg_loss = 0.0, 0.0, 0.0

    with mlflow.start_run():
        mlflow.log_params({**p, "subtask": subtask})

        for epoch in range(p["epochs"]):
            model.train()
            total_loss, train_preds, train_targets = 0.0, [], []

            for inputs, targets in train_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, 1).cpu().tolist())
                train_targets.extend(targets.cpu().tolist())

            avg_loss  = total_loss / len(train_dl)
            train_acc = accuracy_score(train_targets, train_preds)
            train_f1  = f1_score(train_targets, train_preds, average="weighted")

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_dl:
                    preds = torch.argmax(model(inputs.to(device)), 1).cpu().tolist()
                    val_preds.extend(preds)
                    val_targets.extend(targets.tolist())

            val_acc = accuracy_score(val_targets, val_preds)
            val_f1  = f1_score(val_targets, val_preds, average="weighted")

            print(
                f"  Epoch {epoch+1}/{p['epochs']} | Loss: {avg_loss:.4f} "
                f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )
            present = sorted(set(val_targets))
            print(classification_report(val_targets, val_preds, labels=present,
                                        target_names=[labels[i] for i in present]))

            mlflow.log_metrics({"loss": avg_loss, "train_acc": train_acc,
                                 "train_f1": train_f1, "val_acc": val_acc, "val_f1": val_f1},
                                step=epoch)

        mlflow.log_metrics({"final_val_acc": val_acc, "final_val_f1": val_f1})

    out_dir = f"models/lstm_{subtask}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/lstm_model.pt")
    with open(f"{out_dir}/lstm_vocab.pkl", "wb") as f:
        pickle.dump(vocab.word2idx, f)

    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python src/train_lstm.py [a|b|c]")
        sys.exit(1)
    train(subtask)
