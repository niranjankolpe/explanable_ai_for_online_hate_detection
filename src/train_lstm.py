import os
import sys
import yaml
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.pytorch

from dataset_lstm import Vocabulary, OLIDDataset
from model_lstm import LSTMClassifier


TRAIN_PATH = "data/olid-training-v1.0.tsv"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params   = params["lstm"]
VOCAB_SIZE    = lstm_params["vocab_size"]
EMBEDDING_DIM = lstm_params["embedding_dim"]
HIDDEN_DIM    = lstm_params["hidden_dim"]
NUM_LAYERS    = lstm_params["num_layers"]
DROPOUT       = lstm_params["dropout"]
MAX_LEN       = lstm_params["max_len"]
BATCH_SIZE    = lstm_params["batch_size"]
EPOCHS        = lstm_params["epochs"]
LEARNING_RATE = lstm_params["learning_rate"]


def train_subtask(subtask):
    subtask_config = params["subtasks"][subtask]
    column         = subtask_config["column"]
    labels         = subtask_config["labels"]
    num_classes    = len(labels)

    MODEL_PATH = f"models/lstm_{subtask}/lstm_model.pt"
    VOCAB_PATH = f"models/lstm_{subtask}/lstm_vocab.pkl"

    print(f"\nTraining BiLSTM for Subtask {subtask.upper()}...")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"hate_detection_lstm_subtask_{subtask}")
    mlflow.start_run()

    mlflow.log_params({
        "model":         "bilstm",
        "subtask":       subtask,
        "labels":        str(labels),
        "vocab_size":    VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim":    HIDDEN_DIM,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "max_len":       MAX_LEN,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "learning_rate": LEARNING_RATE
    })

    df = pd.read_csv(TRAIN_PATH, sep="\t")
    df = df[df[column].notna()].copy()
    # print(f"Subtask {subtask.upper()} total samples: {len(df)}")

    texts_list  = df["tweet"].astype(str).tolist()
    label2idx   = {label: idx for idx, label in enumerate(labels)}
    labels_list = df[column].map(label2idx).tolist()
    labels_list = df[column].map(label2idx).tolist()
    # print(set(labels_list))  # must show {0, 1}
    # assert len(set(labels_list)) > 1, "Only one class in labels!"

    # --- Train / val split (stratified) ---
    X_train, X_val, y_train, y_val = train_test_split(
        texts_list, labels_list,
        test_size=0.2,
        random_state=42,
        stratify=labels_list
    )
    from collections import Counter
    # print("Val label distribution:", Counter(y_val))
    # print("Train label distribution:", Counter(y_train))
    # print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Build vocab on training data only
    train_texts_series = pd.Series(X_train)
    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build_vocab(train_texts_series)

    train_dataset = OLIDDataset(pd.Series(X_train), pd.Series(y_train), vocab, max_len=MAX_LEN)
    val_dataset   = OLIDDataset(pd.Series(X_val),   pd.Series(y_val),   vocab, max_len=MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,  shuffle=False)

    # Sanity check: peek at one batch
    inputs, targets = next(iter(train_loader))
    # print("Input shape:", inputs.shape)
    # print("Targets:", targets[:10])
    # print("Unique targets in batch:", targets.unique())

    model = LSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    label_counts  = Counter(y_train)
    total         = sum(label_counts.values())
    weights       = [total / (len(label_counts) * label_counts[i]) for i in range(num_classes)]
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    val_acc, val_f1, avg_loss = 0, 0, 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_loss  = 0
        all_preds   = []
        all_targets = []

        for inputs, targets in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

        avg_loss  = total_loss / len(train_loader)
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1  = f1_score(all_targets, all_preds, average="weighted")

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                outputs = model(inputs)
                preds   = torch.argmax(outputs, dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(targets.tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1  = f1_score(val_targets, val_preds, average="weighted")

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} "
            f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        present_labels = sorted(set(val_targets))
        present_names  = [labels[i] for i in present_labels]
        print(classification_report(val_targets, val_preds, labels=present_labels, target_names=present_names))

        mlflow.log_metric("loss",      avg_loss,  step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_f1",  train_f1,  step=epoch)
        mlflow.log_metric("val_acc",   val_acc,   step=epoch)
        mlflow.log_metric("val_f1",    val_f1,    step=epoch)

    mlflow.log_metric("final_loss",    avg_loss)
    mlflow.log_metric("final_val_acc", val_acc)
    mlflow.log_metric("final_val_f1",  val_f1)

    os.makedirs(f"models/lstm_{subtask}", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab.word2idx, f)

    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VOCAB_PATH)
    mlflow.end_run()

    print(f"Subtask {subtask.upper()} training complete.")


def main():
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python train_lstm.py [a|b|c]")
        return
    train_subtask(subtask)


if __name__ == "__main__":
    main()