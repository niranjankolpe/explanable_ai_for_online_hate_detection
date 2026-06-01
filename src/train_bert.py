"""
train_bert.py
DistilBERT fine-tuning for subtask a/b/c.

Usage: python src/train_bert.py [a|b|c]
"""

import os
import sys
import time

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.pytorch

from preprocess import preprocess_common

DATA_PATH    = "data/olid-training-v1.0.tsv"
RANDOM_STATE = 42

with open("params.yaml") as f:
    params = yaml.safe_load(f)

p = params["bert"]


class OLIDDatasetBert(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(subtask: str) -> None:
    cfg       = params["subtasks"][subtask]
    label2idx = {lbl: i for i, lbl in enumerate(cfg["labels"])}

    print(f"\nTraining DistilBERT — Subtask {subtask.upper()}")

    df          = pd.read_csv(DATA_PATH, sep="\t")
    df          = df[df[cfg["column"]].notna()].copy()
    df["tweet"] = df["tweet"].apply(preprocess_common)   # consistent preprocessing
    texts       = df["tweet"].tolist()
    labels      = [label2idx[l] for l in df[cfg["column"]]]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(p["model_name"])
    train_dl  = DataLoader(OLIDDatasetBert(X_train, y_train, tokenizer, p["max_len"]),
                           batch_size=p["batch_size"], shuffle=True)
    val_dl    = DataLoader(OLIDDatasetBert(X_val,   y_val,   tokenizer, p["max_len"]),
                           batch_size=p["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        p["model_name"], num_labels=len(cfg["labels"])
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=p["learning_rate"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"bert_subtask_{subtask}")

    val_acc, val_f1 = 0.0, 0.0

    with mlflow.start_run():
        mlflow.log_params({**p, "subtask": subtask, "preprocessing": "preprocess_common"})

        for epoch in range(p["epochs"]):
            print(f"  Epoch {epoch+1}/{p['epochs']} started at {time.strftime('%H:%M:%S')}")
            model.train()
            total_loss = 0.0

            for i, batch in enumerate(train_dl):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                lbls           = batch["label"].to(device)
                optimizer.zero_grad()
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls).loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % 50 == 0:
                    print(f"    Batch {i+1}/{len(train_dl)} | Loss: {loss.item():.4f} | {time.strftime('%H:%M:%S')}")

            avg_loss = total_loss / len(train_dl)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_dl:
                    out   = model(input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"].to(device))
                    preds = torch.argmax(out.logits, dim=1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(batch["label"].tolist())

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1  = f1_score(all_labels, all_preds, average="weighted")

            print(f"  Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            print(classification_report(all_labels, all_preds, target_names=cfg["labels"]))
            mlflow.log_metrics({"loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1}, step=epoch)

        mlflow.log_metrics({"final_val_acc": val_acc, "final_val_f1": val_f1})

    out_dir = f"models/bert_{subtask}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python src/train_bert.py [a|b|c]")
        sys.exit(1)
    train(subtask)
