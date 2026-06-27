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

from model import BiLSTMClassifier
from dataset import pad_sequence
from preprocess import preprocess_common

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
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        
        model_dir = f"models/bert_{subtask}"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return model, tokenizer

    if model_type == "llama":
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        adapter_path = f"models/llama3.2_3b_lora_hate_speech"
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model on CPU using bfloat16 to save memory
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN")
        )
        
        # Wrap with LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        return model, tokenizer

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Proba functions ───────────────────────────────────────────────────────────

def predict_proba(texts: list, model_type: str, model, aux, subtask: str = None) -> np.ndarray:
    """Returns probability array of shape (n_samples, n_classes)."""

    if model_type == "baseline":
        model, vectorizer = model, aux
        cleaned   = [preprocess_common(t) for t in texts]
        X         = vectorizer.transform(cleaned)
        raw_proba = model.predict_proba(X)
        if subtask is not None:
            # sklearn sorts model.classes_ alphabetically; reorder to match params.yaml label order
            target_labels = _params["subtasks"][subtask]["labels"]
            col_order = [list(model.classes_).index(lbl) for lbl in target_labels]
            return raw_proba[:, col_order]
        return raw_proba

    if model_type == "lstm":
        vocab = aux
        seqs  = []
        for text in texts:
            text = preprocess_common(text)
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

    if model_type == "llama":
        tokenizer = aux
        results = []
        for text in texts:
            # We must use the exact same prompt format as training
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n"
                f"Analyze the following tweet and classify if it contains hate speech or offensive language.\\n"
                f"Tweet: \\\"{text}\\\"\\n"
                f"Output exactly 'OFFENSIVE' or 'NOT OFFENSIVE'.<|eot_id|>\\n"
                f"<|start_header_id|>assistant<|end_header_id|>\\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
                
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if "NOT OFFENSIVE" in decoded.upper() or "NOT" in decoded.upper():
                results.append([1.0, 0.0]) # [NOT, OFF]
            else:
                results.append([0.0, 1.0]) # [NOT, OFF]
        
        return np.array(results)

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Helper ────────────────────────────────────────────────────────────────────

def get_label_conf(proba: np.ndarray, subtask: str = "a") -> tuple:
    """Converts single-sample proba array to (label_str, confidence)."""
    labels     = _params["subtasks"][subtask]["labels"]
    idx        = int(np.argmax(proba))
    return labels[idx], float(proba[idx])
