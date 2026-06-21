"""
tests/test_core.py
Unit tests for preprocessing, dataset utilities, vocabulary,
prediction helpers, monitoring, and faithfulness agreement.

Run: python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tempfile

from preprocess import preprocess_common
from dataset    import Vocabulary, pad_sequence


# ── preprocess_common ─────────────────────────────────────────────────────────
assert preprocess_common("Hello @user http://link.com!!!") == "hello"
assert preprocess_common("  YOU ARE IDIOT  ") == "you are idiot"
assert preprocess_common("") == ""
assert preprocess_common("@mention http://url.com") == ""


# ── pad_sequence ──────────────────────────────────────────────────────────────
assert pad_sequence([1, 2, 3], 5)          == [1, 2, 3, 0, 0]
assert pad_sequence([1, 2, 3, 4, 5, 6], 4) == [1, 2, 3, 4]
assert pad_sequence([], 3)                 == [0, 0, 0]


# ── Vocabulary ────────────────────────────────────────────────────────────────
import pandas as pd
vocab = Vocabulary(max_size=10)
vocab.build_vocab(pd.Series(["hello world", "hello python"]))
assert "<PAD>" in vocab.word2idx
assert "<UNK>" in vocab.word2idx
assert "hello" in vocab.word2idx
assert vocab.word2idx["<PAD>"] == 0
assert vocab.word2idx["<UNK>"] == 1


# ── get_label_conf ────────────────────────────────────────────────────────────
# subtask a labels: ["NOT", "OFF"]  →  idx 0=NOT, idx 1=OFF
from predict import get_label_conf

label, conf = get_label_conf(np.array([0.3, 0.7]), "a")
assert label == "OFF"
assert abs(conf - 0.7) < 1e-6

label, conf = get_label_conf(np.array([0.8, 0.2]), "a")
assert label == "NOT"
assert abs(conf - 0.8) < 1e-6

# subtask b labels: ["TIN", "UNT"]
label, conf = get_label_conf(np.array([0.6, 0.4]), "b")
assert label == "TIN"
assert abs(conf - 0.6) < 1e-6

# subtask c labels: ["IND", "GRP", "OTH"]
label, conf = get_label_conf(np.array([0.1, 0.7, 0.2]), "c")
assert label == "GRP"
assert abs(conf - 0.7) < 1e-6


# ── compute_drift ─────────────────────────────────────────────────────────────
from monitor import compute_drift, get_model_breakdown

# empty records → no drift
d = compute_drift([])
assert d["drift_detected"]    == False
assert d["total_predictions"] == 0
assert d["recent_off_rate"]   == 0.0

# 13 OFF / 20 = 65% > 60% threshold → drift
records = [{"label": "OFF", "model": "baseline_a"}] * 13 + \
          [{"label": "NOT", "model": "baseline_a"}] * 7
d = compute_drift(records)
assert d["drift_detected"]    == True
assert d["total_predictions"] == 20
assert d["recent_off_count"]  == 13

# 10 OFF / 20 = 50% → no drift
records2 = [{"label": "OFF", "model": "baseline_a"}] * 10 + \
           [{"label": "NOT", "model": "baseline_a"}] * 10
assert compute_drift(records2)["drift_detected"] == False

# window = last 20 — older records should not affect recent window
long_records = [{"label": "NOT", "model": "baseline_a"}] * 50 + \
               [{"label": "OFF", "model": "baseline_a"}] * 20
d3 = compute_drift(long_records)
assert d3["drift_detected"]    == True
assert d3["total_predictions"] == 70


# ── get_model_breakdown ───────────────────────────────────────────────────────
records3 = [
    {"model": "baseline_a", "label": "OFF"},
    {"model": "baseline_a", "label": "NOT"},
    {"model": "baseline_a", "label": "OFF"},
    {"model": "lstm_a",     "label": "NOT"},
]
bd = get_model_breakdown(records3)
assert bd["baseline_a"]["total"] == 3
assert bd["baseline_a"]["OFF"]   == 2
assert bd["baseline_a"]["NOT"]   == 1
assert bd["lstm_a"]["total"]     == 1
assert bd["lstm_a"]["NOT"]       == 1
assert get_model_breakdown([])   == {}


# ── log_prediction / load_logs ────────────────────────────────────────────────
import monitor

_orig_log_file  = monitor.LOG_FILE
_tmp_log        = tempfile.mktemp(suffix=".log")
monitor.LOG_FILE = _tmp_log
try:
    assert monitor.load_logs() == []
    monitor.log_prediction("hello world", "baseline_a", "OFF",  0.92)
    monitor.log_prediction("nice day",    "lstm_a",     "NOT",  0.85)
    logs = monitor.load_logs()
    assert len(logs)          == 2
    assert logs[0]["label"]   == "OFF"
    assert logs[0]["model"]   == "baseline_a"
    assert logs[1]["label"]   == "NOT"
    assert logs[1]["text"]    == "nice day"
    assert logs[0]["confidence"] == 0.92
finally:
    if os.path.exists(_tmp_log):
        os.remove(_tmp_log)
    monitor.LOG_FILE = _orig_log_file


# ── compute_agreement ─────────────────────────────────────────────────────────
from faithfulness import compute_agreement

lime6 = {"hello": 0.5, "world": 0.3, "foo": 0.1, "bar": 0.2, "baz": 0.4, "qux": 0.6}
shap5 = {"hello": 0.4, "world": 0.2, "foo": 0.3, "bar": 0.1, "baz": 0.5, "xyz": 0.7}
# 5 of 6 words overlap → 5/6 * 100
assert abs(compute_agreement(lime6, shap5) - (5 / 6 * 100)) < 0.01

# full overlap
shap6 = {"hello": 0.4, "world": 0.2, "foo": 0.3, "bar": 0.1, "baz": 0.5, "qux": 0.7}
assert compute_agreement(lime6, shap6) == 100.0

# zero overlap
shap_none = {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4, "eee": 5, "fff": 6}
assert compute_agreement(lime6, shap_none) == 0.0

# empty dicts
assert compute_agreement({}, {})         == 0.0
assert compute_agreement({"a": 1}, {})   == 0.0


print("All tests passed.")
