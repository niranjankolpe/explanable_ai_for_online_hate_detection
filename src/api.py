from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import load_model, predict, predict_proba
# from .explain import explain_prediction

import traceback

app = FastAPI()

model, vocab = load_model()


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Offensive Language Detection API"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": "Something went wrong"}

@app.post("/predict")
def classify(input: TextInput):
    try:
        label, confidence = predict(input.text, model, vocab)
        # explanation = explain_prediction(input.text,
        #                                 model, vocab,
        #                                 lambda texts: predict_proba(texts, model, vocab))
        return {
            "text": input.text,
            "label": label,
            "confidence": confidence,
            # "explanation": explanation
        }
    except Exception as e:
        print("FULL ERROR TRACE:")
        traceback.print_exc()   # THIS is the key
        raise HTTPException(status_code=500, detail=str(e))