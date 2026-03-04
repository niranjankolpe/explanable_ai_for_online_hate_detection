import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from dataset_lstm import Vocabulary, OLIDDataset
from model_lstm import LSTMClassifier


# Configuration
TRAIN_PATH = "data/olid-training-v1.0.tsv"

MODEL_PATH = "models/lstm_model.pt"
VOCAB_PATH = "models/lstm_vocab.pkl"

MAX_LEN = 50
VOCAB_SIZE = 20000
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def main():

    print("Loading training data...")

    df = pd.read_csv(TRAIN_PATH, sep="\t")

    texts = df["tweet"].astype(str)
    labels = df["subtask_a"]

    print("Building vocabulary...")

    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build_vocab(texts)

    print("Creating dataset...")

    dataset = OLIDDataset(texts, labels, vocab, max_len=MAX_LEN)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Initializing model...")

    model = LSTMClassifier(
        vocab_size=len(vocab.word2idx)
    )

    device = torch.device("cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Starting training...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    print("Saving model...")

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)

    print("Saving vocabulary...")

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab.word2idx, f)

    print("Training complete.")


if __name__ == "__main__":
    main()