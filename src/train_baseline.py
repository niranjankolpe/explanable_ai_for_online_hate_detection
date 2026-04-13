import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow

DATA_PATH    = "data/olid-training-v1.0.tsv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def preprocess_text(text_series):
    return (
        text_series
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.strip()
    )


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hate_detection_baseline")
    mlflow.start_run()

    mlflow.log_param("model",        "tfidf_logistic_regression")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("ngram_range",  "(1,2)")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size",    TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    print("Loading data...")
    df         = load_data(DATA_PATH)
    df["tweet"] = preprocess_text(df["tweet"])
    X          = df["tweet"]
    y          = df["subtask_a"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Vectorizing text...")
    vectorizer    = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)

    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(vectorizer, "models/baseline/tfidf_vectorizer.pkl")

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, "models/baseline/baseline_model.pkl")

    print("Evaluating...")
    preds = model.predict(X_val_tfidf)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average="weighted")

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1_weighted", f1)
    mlflow.end_run()

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds))


if __name__ == "__main__":
    main()