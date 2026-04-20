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

DATA_PATH    = "data/olid-training-v1.0.tsv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

with open("params.yaml") as f:
    params = yaml.safe_load(f)


def load_data(path):
    return pd.read_csv(path, sep="\t")


def preprocess_text(text_series):
    return (
        text_series
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.strip()
    )


def train_subtask(subtask):
    subtask_config = params["subtasks"][subtask]
    column         = subtask_config["column"]
    labels         = subtask_config["labels"]

    print(f"\nTraining baseline for Subtask {subtask.upper()}...")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"hate_detection_baseline_subtask_{subtask}")
    mlflow.start_run()

    mlflow.log_param("model",        "tfidf_logistic_regression")
    mlflow.log_param("subtask",      subtask)
    mlflow.log_param("labels",       str(labels))
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("ngram_range",  "(1,2)")
    mlflow.log_param("class_weight", "balanced")

    df = load_data(DATA_PATH)
    df["tweet"] = preprocess_text(df["tweet"])

    # Filter rows where subtask column is not null
    df = df[df[column].notna()].copy()
    print(f"Subtask {subtask.upper()} training samples: {len(df)}")

    X = df["tweet"]
    y = df[column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    vectorizer    = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)

    os.makedirs(f"models/baseline_{subtask}", exist_ok=True)
    joblib.dump(vectorizer, f"models/baseline_{subtask}/tfidf_vectorizer.pkl")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, f"models/baseline_{subtask}/baseline_model.pkl")

    preds = model.predict(X_val_tfidf)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average="weighted")

    mlflow.log_metric("val_accuracy",    acc)
    mlflow.log_metric("val_f1_weighted", f1)
    mlflow.sklearn.log_model(model, f"baseline_model_subtask_{subtask}")
    mlflow.end_run()

    print(f"Subtask {subtask.upper()} | Acc: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_val, preds))


def main():
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python train_baseline.py [a|b|c]")
        return
    train_subtask(subtask)


if __name__ == "__main__":
    main()