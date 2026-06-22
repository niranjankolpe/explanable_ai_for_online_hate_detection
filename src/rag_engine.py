"""
rag_engine.py
RAG (Retrieval-Augmented Generation) engine for explainable hate detection.

Uses ChromaDB as the vector store, sentence-transformers for embeddings,
and Google Gemini via LangChain for natural language explanations.

Usage:
    from rag_engine import load_vector_store, retrieve_similar, generate_explanation
"""

import os
import time

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


# ── Paths ─────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "models/chroma_store"
COLLECTION_NAME = "olid_tweets"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Embedding model (loaded once) ────────────────────────────────────────────

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Vector store ──────────────────────────────────────────────────────────────

def load_vector_store() -> chromadb.Collection:
    """Load the pre-built ChromaDB collection from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB store not found at '{CHROMA_DIR}'. "
            "Run: python src/build_vector_store.py"
        )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve_similar(text: str, collection: chromadb.Collection, k: int = 5) -> list[dict]:
    """
    Retrieve the K most similar tweets from the vector store.

    Returns a list of dicts with keys: tweet, label_a, label_b, label_c, distance
    """
    embedder   = _get_embedder()
    query_emb  = embedder.encode([text]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    similar = []
    for i in range(len(results["documents"][0])):
        similar.append({
            "tweet":    results["documents"][0][i],
            "label_a":  results["metadatas"][0][i].get("label_a", ""),
            "label_b":  results["metadatas"][0][i].get("label_b", ""),
            "label_c":  results["metadatas"][0][i].get("label_c", ""),
            "distance": results["distances"][0][i],
        })
    return similar


# ── LLM explanation ───────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are an AI explainability assistant for a hate speech detection system \
built on the OLID dataset (Offensive Language Identification Dataset).

A user submitted this text:
\"{text}\"

The {model_name} model classified it as: {prediction} (confidence: {confidence:.1%})

The LIME explainer identified these important words (positive score = pushes toward offensive):
{lime_words}

The SHAP explainer identified these important words (positive score = pushes toward offensive):
{shap_words}

Here are similar tweets from the training data with their known labels:
{similar_examples}

Explain to a non-technical user:
1. Why the model likely made this prediction
2. Which words most influenced the decision and why
3. How confident we should be, given the similar examples

Keep it clear, concise, and under 200 words. Do not use technical jargon."""


def _format_word_scores(scores: dict) -> str:
    """Format a word→score dict into a readable string."""
    if not scores:
        return "(none)"
    parts = []
    for word, score in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "offensive" if score > 0 else "not offensive"
        parts.append(f"  \"{word}\" → {score:+.4f} ({direction})")
    return "\n".join(parts)


def _format_similar(examples: list[dict]) -> str:
    """Format retrieved similar examples into a readable string."""
    if not examples:
        return "(none found)"
    lines = []
    for i, ex in enumerate(examples, 1):
        label = ex["label_a"]
        lines.append(f"  {i}. \"{ex['tweet'][:80]}\" → {label}")
    return "\n".join(lines)


def generate_explanation(
    text: str,
    model_name: str,
    prediction: str,
    confidence: float,
    lime_scores: dict,
    shap_scores: dict,
    similar_examples: list[dict],
    api_key: str = None,
) -> str:
    """
    Generate a natural language explanation using Google Gemini via LangChain.

    Falls back to a structured text summary if no API key is available.
    """
    # Try to get API key from argument, env var, or return fallback
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return _fallback_explanation(
            text, model_name, prediction, confidence,
            lime_scores, shap_scores, similar_examples,
        )

    prompt = PromptTemplate(
        input_variables=[
            "text", "model_name", "prediction", "confidence",
            "lime_words", "shap_words", "similar_examples",
        ],
        template=_PROMPT_TEMPLATE,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=key,
        temperature=0.3,
    )

    chain = prompt | llm

    retries = 3
    delay = 2.0
    for attempt in range(retries):
        try:
            result = chain.invoke({
                "text":             text,
                "model_name":       model_name,
                "prediction":       prediction,
                "confidence":       confidence,
                "lime_words":       _format_word_scores(lime_scores),
                "shap_words":       _format_word_scores(shap_scores),
                "similar_examples": _format_similar(similar_examples),
            })
            return result.content
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = "429" in err_msg or "quota" in err_msg.lower() or "resourceexhausted" in err_msg.lower()
            if is_rate_limit and attempt < retries - 1:
                print(f"[RAG Engine] Rate limit hit (429/ResourceExhausted). Retrying in {delay}s (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                print(f"[RAG Engine] Failed to generate AI explanation: {e}")
                fallback = _fallback_explanation(
                    text, model_name, prediction, confidence,
                    lime_scores, shap_scores, similar_examples,
                )
                return f"*⚠️ Note: Could not generate AI natural language explanation (API error: {e}). Showing structured fallback instead:*\n\n{fallback}"


def _fallback_explanation(
    text, model_name, prediction, confidence,
    lime_scores, shap_scores, similar_examples,
) -> str:
    """Structured summary when no LLM API key is available."""
    lines = [
        f"**Model:** {model_name}",
        f"**Prediction:** {prediction} (confidence: {confidence:.1%})",
        "",
        "**Key words pushing toward this prediction (LIME):**",
        _format_word_scores(lime_scores),
        "",
        "**Key words pushing toward this prediction (SHAP):**",
        _format_word_scores(shap_scores),
        "",
        "**Similar tweets from training data:**",
        _format_similar(similar_examples),
        "",
        "_Set GOOGLE_API_KEY to get a natural language explanation from Gemini._",
    ]
    return "\n".join(lines)
