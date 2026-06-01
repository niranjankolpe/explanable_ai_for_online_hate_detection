"""
faithfulness.py
Computes LIME vs SHAP top-N word agreement across a sample of predictions.

Usage: python src/faithfulness.py [a|b|c]
"""

import gc
import json
import os
import sys
import random

import shap
import pandas as pd
import yaml

from lime.lime_text import LimeTextExplainer
from explain  import explain_with_lime
from predict  import load_model, predict_proba

with open("params.yaml") as f:
    params = yaml.safe_load(f)

SAMPLE_SIZE  = 100
NUM_FEATURES = 6
RANDOM_SEED  = 42
DATA_PATH    = "data/olid-training-v1.0.tsv"


def compute_agreement(lime_dict: dict, shap_dict: dict) -> float:
    lime_words = set(w.lower().strip() for w in lime_dict.keys())
    shap_words = set(w.lower().strip() for w in shap_dict.keys())
    if not lime_words or not shap_words:
        return 0.0
    return len(lime_words & shap_words) / NUM_FEATURES * 100


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
    # for model_type in ["bert"]:
        print(f"  Running {model_type.upper()}...")

        # Load model fresh for each type
        model, aux = load_model(model_type, subtask)

        # Capture in default args to avoid closure issues
        def predict_fn(texts, _m=model, _a=aux, _mt=model_type):
            return predict_proba(texts, _mt, _m, _a)

        # Create SHAP explainer ONCE per model — not per sample
        print(f"  Building SHAP explainer for {model_type}...")
        # Skip SHAP for BERT — too costly on CPU
        if model_type == "bert":
            print("  BERT: SHAP infeasible on CPU — skipping, reporting as N/A")
            results["bert"] = {
                "avg_agreement_pct": None,
                "samples_evaluated": 0,
                "samples_failed":    0,
                "note": "SHAP PartitionExplainer too costly for transformer on CPU"
            }
            del model, aux
            gc.collect()
            continue

        shap_explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text(r"\W+"))

        agreements = []
        failed     = 0

        for i, text in enumerate(sample_texts):
            try:
                lime_exp  = explain_with_lime(text, predict_fn, class_names, NUM_FEATURES)

                shap_vals = shap_explainer([text])
                tokens    = shap_vals.data[0]
                values    = shap_vals.values[0][:, 1]
                top       = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)[:NUM_FEATURES]
                shap_exp  = {t.strip(): float(v) for t, v in top}

                agreements.append(compute_agreement(lime_exp, shap_exp))
            except Exception as e:
                failed += 1
                print(f"    Sample {i+1} failed: {e}")
                continue

            if (i + 1) % 5 == 0:
                avg = sum(agreements) / len(agreements) if agreements else 0
                print(f"    {i+1}/{SAMPLE_SIZE} | Avg agreement: {avg:.2f}%")

        avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
        results[model_type] = {
            "avg_agreement_pct": round(avg_agreement, 2),
            "samples_evaluated": len(agreements),
            "samples_failed":    failed,
        }
        print(f"  {model_type.upper()} | Agreement: {avg_agreement:.2f}% | Failed: {failed}/{SAMPLE_SIZE}\n")

        # Free memory before loading next model
        del model, aux, shap_explainer
        gc.collect()

    os.makedirs("reports", exist_ok=True)
    report = {
        "subtask":      subtask,
        "sample_size":  SAMPLE_SIZE,
        "num_features": NUM_FEATURES,
        "results":      results,
    }
    with open(f"reports/faithfulness_{subtask}.json", "w") as f:
        json.dump(report, f, indent=4)

    print("===== Faithfulness Summary =====")
    print(f"{'Model':12} | {'Agreement (%)':>15} | {'Samples':>8}")
    print("-" * 42)
    for m, v in results.items():
        agr = f"{v['avg_agreement_pct']:.2f}%" if v['avg_agreement_pct'] is not None else "N/A"
        print(f"{m:12} | {agr:>15} | {v['samples_evaluated']:>8}")
    print(f"\nSaved → reports/faithfulness_{subtask}.json")

    return report


if __name__ == "__main__":
    subtask = sys.argv[1] if len(sys.argv) > 1 else "a"
    if subtask not in ["a", "b", "c"]:
        print("Usage: python src/faithfulness.py [a|b|c]")
        sys.exit(1)
    run_faithfulness(subtask)
