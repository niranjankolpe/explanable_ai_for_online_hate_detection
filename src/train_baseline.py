"""
train_baseline.py
TF-IDF + Logistic Regression baseline for subtask a/b/c.

Usage: python src/train_baseline.py [a|b|c]
"""

import os
import sys
import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.sklearn

from preprocess import preprocess_common

DATA_PATH    = "data/olid-training-v1.0.tsv"
RANDOM_STATE = 42

with open("params.yaml") as f:
    params = yaml.safe_load(f)


def train(subtask: str) -> None:
    cfg    = params["subtasks"][subtask]
    column = cfg["column"]

    print(f"\nTraining Baseline — Subtask {subtask.upper()}")

    df           = pd.read_csv(DATA_PATH, sep="\t")
    df["tweet"]  = df["tweet"].apply(preprocess_common)
    df           = df[df[column].notna()].copy()
    print(f"  Samples: {len(df)}")

    X_train, X_val, y_train, y_val = train_test_split(
        df["tweet"], df[column],
        test_size=params["baseline"]["test_size"],
        random_state=RANDOM_STATE,
        stratify=df[column],
    )

    vectorizer    = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_val_tfidf)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average="weighted")

    out_dir = f"models/baseline_{subtask}"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{out_dir}/tfidf_vectorizer.pkl")
    joblib.dump(model,      f"{out_dir}/baseline_model.pkl")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"baseline_subtask_{subtask}")
    with mlflow.start_run():
        mlflow.log_params({"subtask": subtask, "max_features": 10000, "ngram_range": "(1,2)", "preprocessing": "preprocess_common"})
        mlflow.log_metric("val_acc",         acc)
        mlflow.log_metric("val_f1_weighted", f1)
        mlflow.sklearn.log_model(model, "baseline_model")

    print(f"  Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
    print(classification_report(y_val, preds))


if __name__ == "__main__":
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python src/train_baseline.py [a|b|c]")
        sys.exit(1)
    train(subtask)
