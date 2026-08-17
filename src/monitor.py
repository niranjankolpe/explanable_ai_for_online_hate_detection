"""
monitor.py
Logs predictions and detects label drift.

Predictions are persisted to S3 (one object per prediction, under
LOG_PREFIX) so the log survives ECS task redeploys — the local disk a
Fargate task writes to is wiped on every deployment. If S3 is unreachable
(e.g. no AWS credentials in local dev), falls back to the local LOG_FILE
so the app and test suite keep working offline.
"""

import json
import os
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError

LOG_BUCKET = os.environ.get(
    "PREDICTION_LOG_BUCKET", "mys3bucketforhatedetectionproject")
LOG_PREFIX = "prediction-logs/"
LOG_FILE = "logs/predictions.log"
DRIFT_THRESHOLD = 0.6
WINDOW_SIZE = 20

_s3 = boto3.client("s3", region_name="ap-south-1")


def _log_local(entry: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_prediction(
        text: str,
        model: str,
        label: str,
        confidence: float) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "text": text[:100],
        "label": label,
        "confidence": round(confidence, 4),
    }
    try:
        key = f"{LOG_PREFIX}{entry['timestamp']}_{uuid.uuid4().hex}.json"
        _s3.put_object(
            Bucket=LOG_BUCKET,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError):
        _log_local(entry)


def _load_s3_logs() -> list:
    records = []
    try:
        paginator = _s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=LOG_BUCKET, Prefix=LOG_PREFIX):
            for obj in page.get("Contents", []):
                body = _s3.get_object(
                    Bucket=LOG_BUCKET, Key=obj["Key"])["Body"].read()
                try:
                    records.append(json.loads(body))
                except json.JSONDecodeError:
                    continue
    except (BotoCoreError, ClientError):
        pass
    return records


def _load_local_logs() -> list:
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


def load_logs() -> list:
    records = _load_s3_logs() + _load_local_logs()
    records.sort(key=lambda r: r.get("timestamp", ""))
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
