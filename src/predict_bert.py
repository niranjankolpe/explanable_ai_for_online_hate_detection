import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

try:
    from .dataset_lstm import preprocess
except ImportError:
    from dataset_lstm import preprocess

import yaml
with open("params.yaml") as f:
    _params = yaml.safe_load(f)
_BERT_MAX_LEN = _params["bert"]["max_len"]


def load_bert_model(subtask="a"):
    model_dir = f"models/bert_{subtask}"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_bert(text, model, tokenizer, max_len=_BERT_MAX_LEN, subtask="a"):
    labels   = _params["subtasks"][subtask]["labels"]
    text     = preprocess(text)
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs          = model(**encoding)
        probs            = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    label = labels[pred.item()]
    return label, confidence.item()


def predict_bert_proba(texts, model, tokenizer, max_len=_BERT_MAX_LEN):
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encodings)
        probs   = torch.softmax(outputs.logits, dim=1).numpy()
    return probs