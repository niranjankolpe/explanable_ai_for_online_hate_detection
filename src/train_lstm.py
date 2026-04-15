import os
import yaml
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.pytorch

from dataset_lstm import Vocabulary, OLIDDataset
from model_lstm import LSTMClassifier

TRAIN_PATH = "data/olid-training-v1.0.tsv"
MODEL_PATH = "models/lstm/lstm_model.pt"
VOCAB_PATH = "models/lstm/lstm_vocab.pkl"

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


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hate_detection_lstm")
    mlflow.start_run()

    mlflow.log_params({
        "model":         "bilstm",
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

    print("Loading training data...")
    df     = pd.read_csv(TRAIN_PATH, sep="\t")
    texts  = df["tweet"].astype(str)
    labels = df["subtask_a"]

    print("Building vocabulary...")
    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build_vocab(texts)

    print("Creating dataset...")
    dataset    = OLIDDataset(texts, labels, vocab, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing model...")
    model  = LSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    avg_loss = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss  = 0
        all_preds   = []
        all_targets = []

        for inputs, targets in dataloader:
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

        avg_loss = total_loss / len(dataloader)
        acc      = accuracy_score(all_targets, all_preds)
        f1       = f1_score(all_targets, all_preds, average="weighted")

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        mlflow.log_metric("loss",         avg_loss, step=epoch)
        mlflow.log_metric("train_acc",    acc,      step=epoch)
        mlflow.log_metric("train_f1",     f1,       step=epoch)

    mlflow.log_metric("final_loss", avg_loss)

    print("Saving model and vocabulary...")
    os.makedirs("models/lstm", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab.word2idx, f)

    mlflow.pytorch.log_model(model, "lstm_model")
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(
        f"runs:/{run_id}/lstm_model",
        "LSTM_Hate_Model"
    )

    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VOCAB_PATH)

    mlflow.end_run()
    print("Training complete.")


if __name__ == "__main__":
    main()