"""
faithfulness.py
Computes LIME vs SHAP top-N word agreement across a sample of predictions.

Usage: python src/faithfulness.py
"""

import json
import os
import sys
import random

import pandas as pd
import yaml

from explain import explain_with_lime, explain_with_shap, create_lime_explainer, create_shap_explainer
from predict import load_model, predict_proba

with open("params.yaml") as f:
    params = yaml.safe_load(f)

SAMPLE_SIZE  = 100
NUM_FEATURES = 6
LIME_SAMPLES = 300
RANDOM_SEED  = 42
DATA_PATH    = "data/olid-training-v1.0.tsv"


def compute_agreement(lime_dict: dict, shap_dict: dict) -> float:
    """Top-N word overlap between LIME and SHAP as a percentage."""
    lime_words = set(w.lower().strip() for w in lime_dict.keys())
    shap_words = set(w.lower().strip() for w in shap_dict.keys())
    if not lime_words or not shap_words:
        return 0.0
    overlap = len(lime_words & shap_words)
    return overlap / NUM_FEATURES * 100


def run_faithfulness(subtask: str = "a") -> dict:
    print(f"\nFaithfulness Analysis — Subtask {subtask.upper()}")
    print(f"Sample size: {SAMPLE_SIZE} | Top-{NUM_FEATURES} words\n")

    cfg         = params["subtasks"][subtask]
    class_names = cfg["labels"]

    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df[df[cfg["column"]].notna()].copy()

    random.seed(RANDOM_SEED)
    sample_texts = random.sample(df["tweet"].tolist(), min(SAMPLE_SIZE, len(df)))

    results = {}

    for model_type in ["baseline", "lstm", "bert"]:
        print(f"  Running {model_type.upper()}...")
        model, aux = load_model(model_type, subtask)

        def predict_fn(texts):
            return predict_proba(texts, model_type, model, aux, subtask)

        lime_explainer = create_lime_explainer(class_names)
        shap_explainer = create_shap_explainer(predict_fn)

        agreements = []
        failed     = 0

        for i, text in enumerate(sample_texts):
            try:
                lime_exp = explain_with_lime(
                    text, predict_fn, class_names, NUM_FEATURES,
                    num_samples=LIME_SAMPLES, explainer=lime_explainer,
                )
                shap_exp = explain_with_shap(text, predict_fn, NUM_FEATURES, explainer=shap_explainer)
                agreement = compute_agreement(lime_exp, shap_exp)
                agreements.append(agreement)
            except Exception as e:
                failed += 1
                continue

            if (i + 1) % 20 == 0:
                avg_so_far = sum(agreements) / len(agreements) if agreements else 0
                print(f"    {i+1}/{SAMPLE_SIZE} | Avg agreement so far: {avg_so_far:.2f}%")

        avg_agreement  = sum(agreements) / len(agreements) if agreements else 0.0

        results[model_type] = {
            "avg_agreement_pct":  round(avg_agreement, 2),
            "samples_evaluated":  len(agreements),
            "samples_failed":     failed,
        }
        print(f"  {model_type.upper()} | Agreement: {avg_agreement:.2f}% | Failed: {failed}/{SAMPLE_SIZE}\n")

    os.makedirs("reports", exist_ok=True)
    report = {
        "subtask":     subtask,
        "sample_size": SAMPLE_SIZE,
        "num_features": NUM_FEATURES,
        "results":     results
    }
    with open(f"reports/faithfulness_{subtask}.json", "w") as f:
        json.dump(report, f, indent=4)

    print("===== Faithfulness Summary =====")
    print(f"{'Model':12} | {'Agreement (%)':>15} | {'Samples':>8}")
    print("-" * 42)
    for m, v in results.items():
        print(f"{m:12} | {v['avg_agreement_pct']:>14.2f}% | {v['samples_evaluated']:>8}")
    print(f"\nSaved → reports/faithfulness_{subtask}.json")

    return report


if __name__ == "__main__":
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python src/faithfulness.py [a|b|c]")
        sys.exit(1)
    run_faithfulness(subtask)
