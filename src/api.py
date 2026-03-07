from fastapi import FastAPI
from pydantic import BaseModel

from .predict import load_model, predict

app = FastAPI()

model, vocab = load_model()


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Offensive Language Detection API"}


@app.post("/predict")
def classify(input: TextInput):

    label, confidence = predict(input.text, model, vocab)

    return {
        "text": input.text,
        "label": label,
        "confidence": confidence
    }