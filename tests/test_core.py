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

# ── crawler.scrape_text ───────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock
from crawler import scrape_text, scrape_multiple

def _mock_response(html, content_type="text/html"):
    resp = MagicMock()
    resp.text = html
    resp.headers = {"Content-Type": content_type}
    resp.raise_for_status = MagicMock()
    return resp

# Basic paragraph extraction
with patch("crawler.requests.get") as mock_get:
    mock_get.return_value = _mock_response(
        "<html><body><p>This is a test sentence.</p><p>Another good paragraph here.</p></body></html>"
    )
    chunks = scrape_text("https://example.com")
    assert len(chunks) == 2
    assert "This is a test sentence." in chunks[0]
    assert "Another good paragraph here." in chunks[1]

# Strips script and style tags
with patch("crawler.requests.get") as mock_get:
    mock_get.return_value = _mock_response(
        "<html><body><script>var x=1;</script><style>.a{}</style>"
        "<p>Visible text here only.</p></body></html>"
    )
    chunks = scrape_text("https://example.com")
    assert len(chunks) == 1
    assert "Visible text here only." in chunks[0]

# Filters short chunks (< 3 words)
with patch("crawler.requests.get") as mock_get:
    mock_get.return_value = _mock_response(
        "<html><body><p>Hi</p><p>This is long enough to keep.</p></body></html>"
    )
    chunks = scrape_text("https://example.com")
    assert len(chunks) == 1
    assert "long enough" in chunks[0]

# Empty page
with patch("crawler.requests.get") as mock_get:
    mock_get.return_value = _mock_response("<html><body></body></html>")
    chunks = scrape_text("https://example.com")
    assert chunks == []

# Script-only page
with patch("crawler.requests.get") as mock_get:
    mock_get.return_value = _mock_response(
        "<html><body><script>console.log('hello');</script></body></html>"
    )
    chunks = scrape_text("https://example.com")
    assert chunks == []


# ── crawler.scrape_multiple ───────────────────────────────────────────────────

# Mix of success and failure
with patch("crawler.scrape_text") as mock_scrape:
    mock_scrape.side_effect = [
        ["Good text chunk here."],
        Exception("Connection timed out"),
    ]
    results = scrape_multiple(["https://good.com", "https://bad.com"])
    assert results["https://good.com"]["status"] == "ok"
    assert results["https://good.com"]["texts"]  == ["Good text chunk here."]
    assert results["https://bad.com"]["status"]   == "error"
    assert "timed out" in results["https://bad.com"]["error"]

# Empty/whitespace URLs are skipped
with patch("crawler.scrape_text") as mock_scrape:
    mock_scrape.return_value = ["Some text content here."]
    results = scrape_multiple(["https://a.com", "", "  "])
    assert len(results) == 1
    assert "https://a.com" in results


# ── crawler.RecursiveCrawler ──────────────────────────────────────────────────
from crawler import RecursiveCrawler

# Test Robots.txt Parsing & Allowed logic
with patch("crawler.RobotFileParser") as mock_parser_cls:
    mock_parser = MagicMock()
    mock_parser.can_fetch.return_value = True
    mock_parser_cls.return_value = mock_parser
    
    crawler = RecursiveCrawler(max_depth=1, max_pages=2)
    assert crawler.is_allowed("https://allowed.com/path") == True

# Test Link Extraction
crawler = RecursiveCrawler(max_depth=1, max_pages=2)
links = crawler.extract_links(
    "https://example.com/start",
    '<html><body><a href="/about">About</a><a href="https://external.com">External</a></body></html>'
)
assert "https://example.com/about" in links
assert "https://external.com" not in links

# Test crawl loop mock
with patch("crawler.requests.get") as mock_get, \
     patch("crawler.RecursiveCrawler.is_allowed") as mock_allowed:

    mock_allowed.return_value = True

    mock_resp = MagicMock()
    mock_resp.text = '<html><body><p>Crawled content chunk.</p><a href="https://example.com/child">Child</a></body></html>'
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    crawler = RecursiveCrawler(max_depth=1, max_pages=2)
    results = crawler.crawl("https://example.com")
    assert "https://example.com" in results
    assert results["https://example.com"]["status"] == "ok"
    assert "Crawled content chunk." in results["https://example.com"]["texts"]


print("All tests passed.")
