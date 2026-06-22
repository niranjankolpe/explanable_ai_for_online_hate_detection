"""
build_vector_store.py
Builds the ChromaDB vector store from the OLID training dataset.

Embeds all training tweets using sentence-transformers (all-MiniLM-L6-v2)
and stores them with their subtask labels in a persistent ChromaDB collection.

Usage: python src/build_vector_store.py
"""

import os
import sys

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from preprocess import preprocess_common


DATA_PATH       = "data/olid-training-v1.0.tsv"
CHROMA_DIR      = "models/chroma_store"
COLLECTION_NAME = "olid_tweets"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE      = 256


def build():
    print("Loading OLID dataset...")
    df = pd.read_csv(DATA_PATH, sep="\t")
    print(f"  Total rows: {len(df)}")

    # Clean tweets
    df["clean_tweet"] = df["tweet"].apply(preprocess_common)

    # Drop empty after cleaning
    df = df[df["clean_tweet"].str.strip().astype(bool)].copy()
    print(f"  After cleaning: {len(df)} tweets")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Embed in batches
    tweets = df["clean_tweet"].tolist()
    print(f"Embedding {len(tweets)} tweets in batches of {BATCH_SIZE}...")

    all_embeddings = []
    for i in range(0, len(tweets), BATCH_SIZE):
        batch = tweets[i : i + BATCH_SIZE]
        embs  = embedder.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        done = min(i + BATCH_SIZE, len(tweets))
        print(f"  {done}/{len(tweets)} embedded")

    # Create ChromaDB persistent store
    print(f"Saving to ChromaDB at {CHROMA_DIR}...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete collection if it already exists (rebuild)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "OLID training tweets with subtask labels"},
    )

    # Add documents in batches (ChromaDB has its own batch limits)
    for i in range(0, len(tweets), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(tweets))
        batch_df = df.iloc[i:end]

        ids       = [f"tweet_{j}" for j in range(i, end)]
        documents = batch_df["clean_tweet"].tolist()
        embeddings = all_embeddings[i:end]

        metadatas = []
        for _, row in batch_df.iterrows():
            metadatas.append({
                "label_a": str(row.get("subtask_a", "")),
                "label_b": str(row.get("subtask_b", "")),
                "label_c": str(row.get("subtask_c", "")),
                "original": str(row["tweet"])[:200],
            })

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  {end}/{len(tweets)} stored in ChromaDB")

    print(f"\nDone. ChromaDB collection '{COLLECTION_NAME}' saved to '{CHROMA_DIR}'")
    print(f"  Total documents: {collection.count()}")


if __name__ == "__main__":
    build()
