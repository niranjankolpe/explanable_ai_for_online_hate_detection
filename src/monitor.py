"""
monitor.py
Logs predictions and detects label drift.
"""

import json
import os
from datetime import datetime

LOG_FILE = "logs/predictions.log"
DRIFT_THRESHOLD = 0.6
WINDOW_SIZE = 20


def log_prediction(
        text: str,
        model: str,
        label: str,
        confidence: float) -> None:
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "text": text[:100],
        "label": label,
        "confidence": round(confidence, 4),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    records = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_drift(records: list) -> dict:
    if not records:
        return {
            "total_predictions": 0,
            "recent_window": 0,
            "recent_off_count": 0,
            "recent_off_rate": 0.0,
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_detected": False,
        }
    recent = records[-WINDOW_SIZE:]
    off_count = sum(1 for r in recent if r["label"] == "OFF")
    off_rate = off_count / len(recent)
    return {
        "total_predictions": len(records),
        "recent_window": len(recent),
        "recent_off_count": off_count,
        "recent_off_rate": round(off_rate, 4),
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": off_rate > DRIFT_THRESHOLD,
    }


def get_model_breakdown(records: list) -> dict:
    breakdown = {}
    for r in records:
        m = r["model"]
        if m not in breakdown:
            breakdown[m] = {"total": 0}
        breakdown[m]["total"] += 1
        breakdown[m].setdefault(r["label"], 0)
        breakdown[m][r["label"]] += 1
    return breakdown
