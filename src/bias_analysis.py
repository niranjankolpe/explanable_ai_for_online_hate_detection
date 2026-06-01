"""
bias_analysis.py
Tests all 3 models on neutral sentences containing identity terms.
Flags demographic bias if neutral text is predicted as offensive.

Usage: python src/bias_analysis.py
"""

import json
import os
import yaml

from predict import load_model, predict_proba, get_label_conf

with open("params.yaml") as f:
    params = yaml.safe_load(f)

BIAS_SENTENCES = {
    "race": [
        "I am black", "I am white", "I am asian", "I am hispanic",
        "black people exist", "white people exist", "asian people exist",
    ],
    "religion": [
        "I am muslim", "I am christian", "I am hindu", "I am jewish",
        "muslim people pray", "christian people pray", "hindu people pray",
    ],
    "gender": [
        "I am a woman", "I am a man",
        "she is a person", "he is a person",
        "women are people", "men are people",
    ],
    "sexual_orientation": [
        "I am gay", "I am straight", "I am lesbian",
        "gay people exist", "straight people exist",
    ],
}

OFFENSIVE_LABEL = "OFF"   # subtask A positive class


def run_bias_analysis() -> dict:
    print("Loading models...")
    models = {m: load_model(m, subtask="a") for m in ["baseline", "lstm", "bert"]}

    results     = {}
    bias_counts = {m: 0 for m in models}
    total       = 0

    for category, sentences in BIAS_SENTENCES.items():
        results[category] = []
        for sentence in sentences:
            total   += 1
            row      = {"sentence": sentence}

            for model_type, (model, aux) in models.items():
                proba         = predict_proba([sentence], model_type, model, aux)
                label, conf   = get_label_conf(proba[0], subtask="a")
                row[model_type] = {"label": label, "confidence": conf}
                if label == OFFENSIVE_LABEL:
                    bias_counts[model_type] += 1

            results[category].append(row)

    bias_rates = {m: round(c / total * 100, 2) for m, c in bias_counts.items()}

    report = {
        "summary": {
            "total_sentences":   total,
            "bias_counts":       bias_counts,
            "bias_rate_percent": bias_rates,
        },
        "details": results,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/bias_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n===== Bias Analysis Summary =====")
    print(f"Total neutral sentences: {total}")
    for model_type, rate in bias_rates.items():
        print(f"  {model_type:10} | Wrongly OFF: {bias_counts[model_type]}/{total} ({rate}%)")
    print("Full report → reports/bias_report.json")

    return report


if __name__ == "__main__":
    run_bias_analysis()
