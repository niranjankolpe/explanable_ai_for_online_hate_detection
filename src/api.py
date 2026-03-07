from fastapi import FastAPI
from pydantic import BaseModel

from .predict import load_model, predict, predict_proba
from .explain import explain_prediction

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

    explanation = explain_prediction(
        input.text,
        model,
        vocab,
        lambda texts: predict_proba(texts, model, vocab)
    )

    return {
        "text": input.text,
        "label": label,
        "confidence": confidence,
        "explanation": explanation
    }