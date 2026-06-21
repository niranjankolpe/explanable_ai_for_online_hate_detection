"""
preprocess.py
Single source of truth for all text preprocessing.

All three models (Baseline, BiLSTM, DistilBERT) use the same function
to ensure fair and consistent model comparison.
"""

import re


def preprocess_common(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+",   "", text)
    return text.strip()
