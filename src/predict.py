"""
predict.py
Unified prediction interface for all three models.

Usage:
    model, aux = load_model("baseline", subtask="a")
    proba       = predict_proba("some text", "baseline", model, aux)
    label, conf = get_label_conf(proba, subtask="a")
"""

import pickle
import joblib

import torch
import numpy as np
import yaml
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from model   import BiLSTMClassifier
from dataset import pad_sequence
from preprocess import preprocess_common, preprocess_lstm

with open("params.yaml") as f:
    _params = yaml.safe_load(f)

_lstm_p = _params["lstm"]
_bert_p = _params["bert"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model(model_type: str, subtask: str = "a"):
    """
    Returns (model, aux) where aux is:
      baseline → (sklearn_model, vectorizer)
      lstm     → (BiLSTMClassifier, vocab_dict)
      bert     → (DistilBert, tokenizer)
    """
    if model_type == "baseline":
        model      = joblib.load(f"models/baseline_{subtask}/baseline_model.pkl")
        vectorizer = joblib.load(f"models/baseline_{subtask}/tfidf_vectorizer.pkl")
        return model, vectorizer

    if model_type == "lstm":
        with open(f"models/lstm_{subtask}/lstm_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        labels     = _params["subtasks"][subtask]["labels"]
        model = BiLSTMClassifier(
            vocab_size=max(vocab.values()) + 1,
            embedding_dim=_lstm_p["embedding_dim"],
            hidden_dim=_lstm_p["hidden_dim"],
            num_layers=_lstm_p["num_layers"],
            dropout=_lstm_p["dropout"],
            num_classes=len(labels),
        )
        model.load_state_dict(
            torch.load(f"models/lstm_{subtask}/lstm_model.pt", map_location="cpu")
        )
        model.eval()
        return model, vocab

    if model_type == "bert":
        model_dir = f"models/bert_{subtask}"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return model, tokenizer

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Proba functions ───────────────────────────────────────────────────────────

def predict_proba(texts: list, model_type: str, model, aux) -> np.ndarray:
    """Returns probability array of shape (n_samples, n_classes)."""

    if model_type == "baseline":
        model, vectorizer = model, aux
        cleaned  = [preprocess_common(t) for t in texts]
        X        = vectorizer.transform(cleaned)
        return model.predict_proba(X)

    if model_type == "lstm":
        vocab = aux
        seqs  = []
        for text in texts:
            text = preprocess_lstm(text)
            seq  = [vocab.get(t, vocab["<UNK>"]) for t in text.split()] if text else [vocab["<UNK>"]]
            seqs.append(pad_sequence(seq, _lstm_p["max_len"]))
        inputs = torch.tensor(seqs)
        with torch.no_grad():
            return torch.softmax(model(inputs), dim=1).numpy()

    if model_type == "bert":
        tokenizer = aux
        cleaned   = [preprocess_common(t) for t in texts]
        enc = tokenizer(
            cleaned,
            max_length=_bert_p["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, dim=1).numpy()

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Helper ────────────────────────────────────────────────────────────────────

def get_label_conf(proba: np.ndarray, subtask: str = "a") -> tuple:
    """Converts single-sample proba array to (label_str, confidence)."""
    labels     = _params["subtasks"][subtask]["labels"]
    idx        = int(np.argmax(proba))
    return labels[idx], float(proba[idx])
