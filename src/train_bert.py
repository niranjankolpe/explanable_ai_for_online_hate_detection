import os
import yaml
import pandas as pd
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim import AdamW

import mlflow
import mlflow.pytorch

TRAIN_PATH = "data/olid-training-v1.0.tsv"
MODEL_DIR  = "models/bert"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

bert_params   = params["bert"]
MAX_LEN       = bert_params["max_len"]
BATCH_SIZE    = bert_params["batch_size"]
EPOCHS        = bert_params["epochs"]
LEARNING_RATE = bert_params["learning_rate"]
MODEL_NAME    = bert_params["model_name"]


class OLIDDatasetBert(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hate_detection_bert")
    mlflow.start_run()

    mlflow.log_params({
        "model":         MODEL_NAME,
        "max_len":       MAX_LEN,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "learning_rate": LEARNING_RATE
    })

    print("Loading data...")
    df     = pd.read_csv(TRAIN_PATH, sep="\t")
    texts  = df["tweet"].astype(str).tolist()
    labels = [1 if l == "OFF" else 0 for l in df["subtask_a"]]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = OLIDDatasetBert(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset   = OLIDDatasetBert(X_val,   y_val,   tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    print(f"Loading model: {MODEL_NAME}")
    model  = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting epochs...at {time.ctime()}")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} of {EPOCHS} at {time.ctime()}")
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            print(f"Iteration: {i} at {time.ctime()}")
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            optimizer.zero_grad()
            outputs    = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss       = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Getting Average loss at {time.ctime()}")
        avg_loss = total_loss / len(train_loader)

        print(f"Starting model validation at {time.ctime()}")
        model.eval()
        all_preds, all_labels = [], []

        print(f"Loading batches at {time.ctime()}")
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                print(f"Batch: {i} at {time.ctime()}")
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
                preds          = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].tolist())
        
        print(f"Getting metrics at {time.ctime()}")
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average="weighted")

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["NOT", "OFF"]))

        mlflow.log_metric("loss",     avg_loss, step=epoch)
        mlflow.log_metric("val_acc",  val_acc,  step=epoch)
        mlflow.log_metric("val_f1",   val_f1,   step=epoch)

    mlflow.log_metric("final_val_acc", val_acc)
    mlflow.log_metric("final_val_f1",  val_f1)

    print("Saving model and tokenizer...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    mlflow.pytorch.log_model(model, "bert_model")
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(
        f"runs:/{run_id}/bert_model",
        "BERT_Hate_Model"
    )

    mlflow.log_artifacts(MODEL_DIR)
    
    mlflow.end_run()
    print("Done.")


if __name__ == "__main__":
    main()