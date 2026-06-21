# Explainable AI Framework for Online Hate Detection

An end-to-end NLP framework for detecting offensive language and explaining model decisions using LIME and SHAP. Built on the OLID dataset (SemEval 2019 Task 6) with a full MLOps stack.

---

## Project Overview

| Component | Details |
|---|---|
| Dataset | OLID v1.0 — 13,240 training tweets (SemEval 2019) |
| Task | Cascading subtask classification: Offensive detection → Type → Target |
| Models | Baseline (TF-IDF + LR), BiLSTM, DistilBERT |
| Explainability | LIME + SHAP word attribution, faithfulness analysis |
| Responsible AI | Bias detection across race, religion, gender, sexual orientation |
| MLOps | DVC, MLflow, Docker, GitHub Actions CI |

---

## Task Hierarchy

```
Subtask A: OFF vs NOT          (all tweets)
    └── Subtask B: TIN vs UNT  (if A = OFF)
            └── Subtask C: IND vs GRP vs OTH  (if B = TIN)
```

---

## Models

| Model | Architecture |
|---|---|
| **Baseline** | TF-IDF (max 20k features, bigrams) + Logistic Regression |
| **BiLSTM** | 2-layer bidirectional LSTM, 128-dim embeddings, 128 hidden, dropout 0.5, 15 epochs |
| **DistilBERT** | `distilbert-base-uncased` fine-tuned, 3 epochs, lr 2e-5, batch 32 |

---

## Results

### Subtask A — Offensive vs Not Offensive (860 test samples)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline | 0.765 | 0.760 |
| BiLSTM | 0.690 | 0.697 |
| **DistilBERT** | **0.810** | **0.815** |

### Subtask B — Targeted vs Untargeted (240 test samples)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline | 0.850 | 0.863 |
| BiLSTM | 0.679 | 0.728 |
| **DistilBERT** | — | — |

### Subtask C — Target Identification: IND / GRP / OTH (213 test samples)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline | 0.648 | 0.636 |
| BiLSTM | 0.507 | 0.506 |
| **DistilBERT** | **0.685** | **0.637** |

---

## Explainability

Both LIME and SHAP are applied to the Subtask A classifier for any input text.

- **LIME** — locally approximates the model with a linear surrogate; identifies which words push toward OFF or NOT
- **SHAP** — Shapley values assign each word a contribution score based on cooperative game theory
- **Faithfulness** — measures top-N word overlap between LIME and SHAP across 100 sampled predictions

### Faithfulness Results (LIME vs SHAP top-6 word agreement)

| Model | Subtask A | Subtask B | Subtask C |
|---|---|---|---|
| Baseline | 69.00% | 67.67% | 69.67% |
| BiLSTM | 54.17% | 56.67% | 63.83% |
| DistilBERT | — (CPU limit) | — (CPU limit) | — (CPU limit) |

> DistilBERT SHAP (PartitionExplainer) is too costly on CPU. Run on GPU for BERT faithfulness scores.

---

## Bias Analysis

Tests all 3 models on 25 neutral sentences containing identity terms across 4 categories: race, religion, gender, sexual orientation. Flagging neutral text as offensive indicates demographic bias.

| Model | Biased Predictions | Bias Rate |
|---|---|---|
| Baseline | 6 / 25 | 24.0% |
| BiLSTM | 7 / 25 | 28.0% |
| DistilBERT | 5 / 25 | 20.0% |

---

## Monitoring & Drift Detection

All predictions are logged to `logs/predictions.log`. Drift is flagged when the offensive rate in the last 20 predictions exceeds 60%.

---

## MLOps Stack

| Tool | Role |
|---|---|
| **DVC** | Data and model versioning, reproducible pipeline (`dvc repro`) |
| **MLflow** | Experiment tracking, metrics, params (SQLite backend) |
| **Docker** | Containerised Streamlit app (`python:3.10-slim`) |
| **GitHub Actions** | CI: flake8 lint + unit tests on every push to main |

### DVC Pipeline

```
train_baseline@[a,b,c]  ──┐
train_lstm@[a,b,c]      ──┼──► evaluate ──► reports/metrics.json
train_bert@[a,b,c]      ──┘
                           └──► bias_analysis ──► reports/bias_report.json
                           └──► faithfulness@[a,b,c] ──► reports/faithfulness_*.json
```

Run the full pipeline:
```bash
dvc repro
```

---

## Installation

**System Requirements** (Windows 11)
- Python 3.10.x
- Git 2.52.0 (Windows)
- DVC 3.65.0
- 8 GB+ RAM

**Steps**

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd <project-dir>
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt --force-reinstall --no-cache-dir

# 2. Configure DVC local remote (Git Bash)
dvc remote add -d storage C:/dvc-storage --local

# 3. Pull trained models or reproduce from scratch
dvc pull       # if DVC remote was configured on this machine before
dvc repro      # to retrain everything from scratch
```

---

## Running the App

```bash
streamlit run src/app.py
```

Open the local URL in your browser. The app has 4 tabs:

| Tab | Description |
|---|---|
| **Prediction** | Cascading classification through Subtask A → B → C |
| **Explanation** | LIME + SHAP word importance charts and side-by-side comparison |
| **Bias Analysis** | Demographic bias report across identity categories |
| **Monitoring** | Prediction log, drift detection, per-model breakdown |

**Sample inputs**
```
"you are idiot"   → OFF
"you are nice"    → NOT
"nice work bro"   → NOT
```

---

## Docker

```bash
# Build (uses requirements-docker.txt — OS-specific deps removed)
docker build -t hate-detection-app .

# Run
docker run -p 8501:8501 hate-detection-app
```

> `requirements-docker.txt` mirrors `requirements.txt` with Windows-only packages (e.g. `pywin32`) removed.

---

## Testing

```bash
python tests/test_core.py
```

Covers: `preprocess_common`, `pad_sequence`, `Vocabulary`, `get_label_conf`, `compute_drift`, `get_model_breakdown`, `log_prediction`, `load_logs`, `compute_agreement`.

---

## Project Structure

```
├── data/                  # OLID dataset (DVC-tracked)
├── models/                # Trained models (DVC-tracked)
│   ├── baseline_[a,b,c]/
│   ├── lstm_[a,b,c]/
│   └── bert_[a,b,c]/
├── reports/               # metrics.json, bias_report.json, faithfulness_*.json
├── src/
│   ├── app.py             # Streamlit UI
│   ├── train_baseline.py  # TF-IDF + LR training
│   ├── train_lstm.py      # BiLSTM training
│   ├── train_bert.py      # DistilBERT fine-tuning
│   ├── predict.py         # Unified predict_proba interface
│   ├── explain.py         # LIME + SHAP explanations
│   ├── faithfulness.py    # LIME vs SHAP agreement analysis
│   ├── evaluate.py        # Test set evaluation + MLflow logging
│   ├── bias_analysis.py   # Demographic bias detection
│   ├── monitor.py         # Prediction logging + drift detection
│   ├── preprocess.py      # Text cleaning
│   ├── dataset.py         # Vocabulary + dataset utilities
│   └── model.py           # BiLSTM architecture
├── tests/
│   └── test_core.py
├── dvc.yaml               # Pipeline definition
├── params.yaml            # Centralised hyperparameters
├── Dockerfile
└── .github/workflows/ci.yml
```
