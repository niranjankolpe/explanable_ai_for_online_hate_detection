import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report

import os
import json

# Configuration
MODEL_PATH      = "models/baseline_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
TEST_PATH       = "data/testset-levela.tsv"
LABELS_PATH     = "data/labels-levela.csv"


# Load Test Data
def load_test_data():
    X_test = pd.read_csv(TEST_PATH, sep="\t")
    y_test = pd.read_csv(LABELS_PATH, header=None, names=["id", "label"])

    # Merge labels with test set
    df = X_test.merge(y_test, on="id")
    return df


# Main Evaluation
def main():

    print("Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading test data...")
    df = load_test_data()

    X = df["tweet"].str.lower()
    y_true = df["label"]

    print("Vectorizing...")
    X_tfidf = vectorizer.transform(X)

    print("Predicting...")
    preds = model.predict(X_tfidf)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    # Save metrics
    os.makedirs("reports", exist_ok=True)

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1)
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

# Driver code
if __name__ == "__main__":
    main()
