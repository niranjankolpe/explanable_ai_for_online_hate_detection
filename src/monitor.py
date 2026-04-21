import json
import os
from datetime import datetime

PREDICTIONS_LOG = "logs/predictions.log"
DRIFT_THRESHOLD = 0.6  # Alert if OFF rate exceeds 60% in recent window
WINDOW_SIZE     = 20   # Number of recent predictions to check for drift


def log_prediction(text, model, label, confidence):
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp":  datetime.now().isoformat(),
        "model":      model,
        "text":       text[:100],
        "label":      label,
        "confidence": round(confidence, 4)
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs():
    if not os.path.exists(PREDICTIONS_LOG):
        return []
    with open(PREDICTIONS_LOG, "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_drift(records):
    if not records:
        return None

    recent   = records[-WINDOW_SIZE:]
    off_count = sum(1 for r in recent if r["label"] == "OFF")
    off_rate  = off_count / len(recent)
    drift     = off_rate > DRIFT_THRESHOLD

    return {
        "total_predictions":  len(records),
        "recent_window":      len(recent),
        "recent_off_count":   off_count,
        "recent_off_rate":    round(off_rate, 4),
        "drift_threshold":    DRIFT_THRESHOLD,
        "drift_detected":     drift
    }


def get_model_breakdown(records):
    breakdown = {}
    for r in records:
        m = r["model"]
        if m not in breakdown:
            breakdown[m] = {"total": 0}
        breakdown[m]["total"] += 1
        label = r["label"]
        if label not in breakdown[m]:
            breakdown[m][label] = 0
        breakdown[m][label] += 1
    return breakdown