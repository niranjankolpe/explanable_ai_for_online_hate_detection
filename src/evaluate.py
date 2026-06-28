"""
evaluate.py
Evaluates all three models on official OLID test sets.

Usage: python src/evaluate.py [a|b|c]   (omit subtask to run all)
"""

import os
import sys
import json

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow

from predict import load_model, predict_proba

with open("params.yaml") as f:
    params = yaml.safe_load(f)


def load_test_data(subtask: str) -> pd.DataFrame:
    cfg = params["subtasks"][subtask]
    X_test = pd.read_csv(cfg["test_file"], sep="\t")
    y_test = pd.read_csv(
        cfg["labels_file"],
        header=None,
        names=[
            "id",
            "label"])
    df = X_test.merge(y_test, on="id")
    return df[df["label"].notna()].copy()


def evaluate_model(df: pd.DataFrame, model_type: str, subtask: str):
    model_dir = f"models/{model_type}_{subtask}"
    if not os.path.exists(model_dir):
        print(
            f"  Skipping {model_type.upper()} — Subtask {subtask.upper()} (model not found)")
        return None

    print(f"  Evaluating {model_type.upper()} — Subtask {subtask.upper()}...")

    cfg = params["subtasks"][subtask]
    labels = cfg["labels"]
    label2idx = {lbl: i for i, lbl in enumerate(labels)}

    model, aux = load_model(model_type, subtask)
    texts = df["tweet"].tolist()
    y_true = df["label"].map(label2idx).tolist()

    proba = predict_proba(texts, model_type, model, aux, subtask)
    preds = proba.argmax(axis=1).tolist()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")
    print(classification_report(y_true, preds, target_names=labels))
    return {"accuracy": float(acc), "f1_weighted": float(f1)}


def evaluate_subtask(subtask: str) -> dict:
    print(f"\n{'='*45}")
    print(f" Subtask {subtask.upper()}")
    print(f"{'='*45}")

    df = load_test_data(subtask)
    results = {}
    for model_type in ["baseline", "lstm", "bert"]:
        r = evaluate_model(df, model_type, subtask)
        if r is not None:
            results[model_type] = r

    print("\n  Results:")
    for m, v in results.items():
        print(
            f"  {m:10} | Acc: {v['accuracy']:.4f} | F1: {v['f1_weighted']:.4f}")

    return results


def main():
    subtask_arg = sys.argv[1] if len(sys.argv) > 1 else None
    subtasks = [subtask_arg] if subtask_arg in [
        "a", "b", "c"] else [
        "a", "b", "c"]
    all_metrics = {f"subtask_{st}": evaluate_subtask(st) for st in subtasks}

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("evaluation")
    with mlflow.start_run():
        for st, metrics in all_metrics.items():
            for model, vals in metrics.items():
                mlflow.log_metric(f"{st}_{model}_accuracy", vals["accuracy"])
                mlflow.log_metric(
                    f"{st}_{model}_f1_weighted",
                    vals["f1_weighted"])

    print("\nMetrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()
