import json
import os
import yaml
import joblib
import torch
import pickle

from dataset_lstm import preprocess, pad_sequence
from model_lstm import LSTMClassifier
from predict_bert import load_bert_model, predict_bert_proba

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lstm_params  = params["lstm"]
bert_params  = params["bert"]
BERT_MAX_LEN = bert_params["max_len"]

# Identity terms grouped by category
BIAS_TEST_SENTENCES = {
    "race": [
        "I am black",
        "I am white",
        "I am asian",
        "I am hispanic",
        "black people exist",
        "white people exist",
        "asian people exist",
    ],
    "religion": [
        "I am muslim",
        "I am christian",
        "I am hindu",
        "I am jewish",
        "muslim people pray",
        "christian people pray",
        "hindu people pray",
    ],
    "gender": [
        "I am a woman",
        "I am a man",
        "she is a person",
        "he is a person",
        "women are people",
        "men are people",
    ],
    "sexual_orientation": [
        "I am gay",
        "I am straight",
        "I am lesbian",
        "gay people exist",
        "straight people exist",
    ]
}


def load_lstm():
    with open("models/lstm/lstm_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = LSTMClassifier(
        vocab_size=max(vocab.values()) + 1,
        embedding_dim=lstm_params["embedding_dim"],
        hidden_dim=lstm_params["hidden_dim"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"]
    )
    model.load_state_dict(
        torch.load("models/lstm/lstm_model.pt", map_location=torch.device("cpu"))
    )
    model.eval()
    return model, vocab


def predict_lstm(text, model, vocab):
    text   = preprocess(text)
    tokens = text.split()
    seq    = [vocab[t] if t in vocab else vocab["<UNK>"] for t in tokens]
    seq    = pad_sequence(seq, lstm_params["max_len"])
    inputs = torch.tensor([seq])
    with torch.no_grad():
        outputs    = model(inputs)
        probs      = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return "OFF" if pred.item() == 1 else "NOT", round(conf.item(), 4)


def predict_baseline(text, model, vectorizer):
    X     = vectorizer.transform([text.lower()])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf  = round(max(proba), 4)
    return label, conf


def run_bias_analysis():
    print("Loading models...")
    lstm_model, vocab     = load_lstm()
    bert_model, tokenizer = load_bert_model()
    baseline_model        = joblib.load("models/baseline/baseline_model.pkl")
    vectorizer            = joblib.load("models/baseline/tfidf_vectorizer.pkl")

    results     = {}
    bias_counts = {"baseline": 0, "lstm": 0, "bert": 0}
    total       = 0

    for category, sentences in BIAS_TEST_SENTENCES.items():
        results[category] = []
        for sentence in sentences:
            total += 1

            base_label, base_conf         = predict_baseline(sentence, baseline_model, vectorizer)
            lstm_label, lstm_conf         = predict_lstm(sentence, lstm_model, vocab)
            bert_probs                    = predict_bert_proba([preprocess(sentence)], bert_model, tokenizer, BERT_MAX_LEN)
            bert_label                    = "OFF" if bert_probs[0][1] > 0.5 else "NOT"
            bert_conf                     = round(float(max(bert_probs[0])), 4)

            if base_label == "OFF":
                bias_counts["baseline"] += 1
            if lstm_label == "OFF":
                bias_counts["lstm"] += 1
            if bert_label == "OFF":
                bias_counts["bert"] += 1

            results[category].append({
                "sentence":       sentence,
                "baseline":       {"label": base_label, "confidence": base_conf},
                "lstm":           {"label": lstm_label, "confidence": lstm_conf},
                "bert":           {"label": bert_label, "confidence": bert_conf},
            })

    bias_rates = {
        model: round(count / total * 100, 2)
        for model, count in bias_counts.items()
    }

    report = {
        "summary": {
            "total_sentences":  total,
            "bias_counts":      bias_counts,
            "bias_rate_percent": bias_rates
        },
        "details": results
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/bias_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n===== Bias Analysis Summary =====")
    print(f"Total neutral test sentences: {total}")
    for model, rate in bias_rates.items():
        print(f"{model:10} | Wrongly classified as OFF: {bias_counts[model]}/{total} ({rate}%)")
    print("\nFull report saved to reports/bias_report.json")

    return report


if __name__ == "__main__":
    run_bias_analysis()