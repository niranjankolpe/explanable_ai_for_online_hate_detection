# Importing required libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuration
DATA_PATH = "olid-training-v1.0.tsv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Data Loading
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df

# Preprocessing
def preprocess_text(text_series: pd.Series) -> pd.Series:
    return (
        text_series
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.strip()
    )

# Main Training Pipeline
def train_model():

    print("Loading data...")
    df = load_data(DATA_PATH)

    # Level A classification (OFF vs NOT)
    df["tweet"] = preprocess_text(df["tweet"])
    X = df["tweet"]
    y = df["subtask_a"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    preds = model.predict(X_val_tfidf)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds))

if __name__ == "__main__":
    train_model()
