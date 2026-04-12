import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

try:
    from .dataset_lstm import preprocess
except ImportError:
    from dataset_lstm import preprocess

MODEL_DIR = "models/bert"

def load_bert_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def predict_bert(text, model, tokenizer, max_len=128):
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
    label = "OFF" if pred.item() == 1 else "NOT"
    return label, confidence.item()


def predict_bert_proba(texts, model, tokenizer, max_len=128):
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