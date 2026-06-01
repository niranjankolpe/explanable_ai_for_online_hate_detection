"""
preprocess.py
Single source of truth for all text preprocessing.

preprocess_common : for Baseline + DistilBERT  (lowercase, strip URLs, @mentions)
preprocess_lstm   : for BiLSTM only            (common + remove non-alpha chars)
"""

import re


def preprocess_common(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+",   "", text)
    return text.strip()


def preprocess_lstm(text: str) -> str:
    text = preprocess_common(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()
