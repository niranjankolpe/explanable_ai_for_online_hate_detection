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
| RAG Explainer | LangChain + ChromaDB + Google Gemini for natural language explanations |
| Data Collection | BeautifulSoup web crawler with bulk URL support |
| Responsible AI | Bias detection across race, religion, gender, sexual orientation |
| Live Streaming | Real-time Bluesky firehose moderation via WebSockets |
| Agentic AI | CrewAI multi-agent website audit (Scraper + Analyst agents) |
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

## RAG Explainer

The RAG (Retrieval-Augmented Generation) Explainer uses **LangChain + ChromaDB + Google Gemini** to generate natural language explanations of model predictions.

**How it works:**
1. User submits text → model predicts OFF or NOT
2. LIME and SHAP identify the most important words
3. ChromaDB retrieves the 5 most similar tweets from the OLID training data (using sentence-transformers embeddings)
4. All context is sent to Google Gemini via LangChain, which returns a plain-English explanation

| Component | Details |
|---|---|
| Vector Store | ChromaDB (persistent, 13,240 embedded tweets) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| LLM | Google Gemini (`gemini-2.0-flash`) via LangChain |
| Fallback | Structured text summary if no API key is set |

### Setup

```bash
# 1. Build the vector store (one-time, ~2-3 min on CPU)
python src/build_vector_store.py

# 2. Set your Google API key (get one free at https://aistudio.google.com/apikey)
set GOOGLE_API_KEY=your_key_here    # Windows
```

Alternatively, paste your API key directly in the app's RAG Explainer tab.

---

## Web Data Collection

The Data Collection tab scrapes text from web pages using **BeautifulSoup + requests** and runs hate speech detection on the collected content.

- Supports **bulk URL** input (one URL per line)
- Extracts visible text (paragraphs, headings, list items) while stripping scripts, styles, and navigation
- Runs batch predictions through the selected model
- Displays summary statistics and detailed results
- Saves raw crawled data to `crawled_data/` as timestamped CSV
- Download button for prediction results as CSV

---

## Monitoring & Drift Detection

All predictions are logged to `logs/predictions.log`. Drift is flagged when the offensive rate in the last 20 predictions exceeds 60%.

---

## Live Stream Moderation

Connects to the **Bluesky Jetstream public firehose** via WebSockets for real-time moderation of live social media posts.

- Streams English-language posts from Bluesky's decentralized network
- Runs hate speech classification on each post in real-time
- Displays a scrolling moderated feed with color-coded labels (🟢 NOT / 🔴 OFF)
- Configurable capture count (5–50 posts per session)
- No API key needed — uses the public Jetstream endpoint

---

## Agentic Analysis (CrewAI)

A **multi-agent AI workflow** using CrewAI that performs automated website content auditing.

**How it works:**
1. **Scraper Agent** — Recursively crawls a target website (respecting `robots.txt`), extracting all visible text content
2. **Analyst Agent** — Classifies every text chunk using the local hate speech model and generates a compliance audit report

| Component | Details |
|---|---|
| Framework | CrewAI multi-agent orchestration |
| LLM | Google Gemini (configurable: `gemini-3.5-flash`, `gemini-2.0-flash`) |
| Crawler | Recursive with `robots.txt` compliance, same-domain restriction |
| Classifier | Local baseline model (no API calls for classification) |

> **Note:** Requires a Google Gemini API key. The free tier allows 20 requests/day per model. Each audit run uses ~10-20 API calls.

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

# 4. Build the RAG vector store (one-time)
python src/build_vector_store.py

# 5. (Optional) Set Google API key for RAG explanations
set GOOGLE_API_KEY=your_key_here
```

---

## Running the App

```bash
streamlit run src/app.py
```

Open the local URL in your browser. The app has 8 tabs:

| Tab | Description |
|---|---|
| **Prediction** | Cascading classification through Subtask A → B → C |
| **Explanation** | LIME + SHAP word importance charts and side-by-side comparison |
| **RAG Explainer** | LangChain + ChromaDB + Gemini natural language explanations |
| **Bias Analysis** | Demographic bias report across identity categories |
| **Data Collection** | Scrape text from URLs and run batch hate speech detection |
| **Monitoring** | Prediction log, drift detection, per-model breakdown |
| **Live Stream Moderation** | Real-time Bluesky firehose WebSocket moderation |
| **Agentic Analysis** | CrewAI multi-agent recursive website audit |

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

Covers: `preprocess_common`, `pad_sequence`, `Vocabulary`, `get_label_conf`, `compute_drift`, `get_model_breakdown`, `log_prediction`, `load_logs`, `compute_agreement`, `scrape_text`, `scrape_multiple`.

---

## Project Structure

```
├── data/                       # OLID dataset (DVC-tracked)
│   └── crawled/                # Web-scraped data (CSV files)
├── models/                     # Trained models (DVC-tracked)
│   ├── baseline_[a,b,c]/
│   ├── lstm_[a,b,c]/
│   ├── bert_[a,b,c]/
│   └── chroma_store/           # ChromaDB vector store for RAG
├── reports/                    # metrics.json, bias_report.json, faithfulness_*.json
├── src/
│   ├── app.py                  # Streamlit UI (8 tabs)
│   ├── train_baseline.py       # TF-IDF + LR training
│   ├── train_lstm.py           # BiLSTM training
│   ├── train_bert.py           # DistilBERT fine-tuning
│   ├── predict.py              # Unified predict_proba interface
│   ├── explain.py              # LIME + SHAP explanations
│   ├── faithfulness.py         # LIME vs SHAP agreement analysis
│   ├── evaluate.py             # Test set evaluation + MLflow logging
│   ├── bias_analysis.py        # Demographic bias detection
│   ├── monitor.py              # Prediction logging + drift detection
│   ├── rag_engine.py           # LangChain + ChromaDB RAG pipeline
│   ├── build_vector_store.py   # ChromaDB vector store builder
│   ├── crawler.py              # BeautifulSoup web scraper + recursive crawler
│   ├── agents_workflow.py      # CrewAI multi-agent audit workflow
│   ├── preprocess.py           # Text cleaning
│   ├── dataset.py              # Vocabulary + dataset utilities
│   └── model.py                # BiLSTM architecture
├── tests/
│   └── test_core.py
├── dvc.yaml                    # Pipeline definition
├── params.yaml                 # Centralised hyperparameters
├── Dockerfile
└── .github/workflows/ci.yml
```
