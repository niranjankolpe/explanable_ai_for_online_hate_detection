import logging
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .predict import load_model, predict, predict_proba
from .predict_bert import load_bert_model, predict_bert, predict_bert_proba
from .explain import explain_prediction

import yaml

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

BERT_MAX_LEN = params["bert"]["max_len"]

app = FastAPI(title="Offensive Language Detection API")

# Load models at startup
logger.info("Loading LSTM model...")
lstm_model, vocab = load_model()

logger.info("Loading DistilBERT model...")
bert_model, tokenizer = load_bert_model()

logger.info("All models loaded.")


class TextInput(BaseModel):
    text: str
    model: str = "lstm"   # "lstm" or "bert"


@app.get("/")
def home():
    return {"message": "Offensive Language Detection API", "models": ["lstm", "bert"]}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong"}
    )


@app.post("/predict")
def classify(input: TextInput):
    try:
        logger.info(f"Request | model={input.model} | text={input.text[:80]}")

        if input.model == "bert":
            print("Inside bert...")
            label, confidence = predict_bert(input.text, bert_model, tokenizer, BERT_MAX_LEN)
            print(f"Got label: {label}, confidence: {confidence}")
            # explanation = explain_prediction(
            #     input.text,
            #     lambda texts: predict_bert_proba(texts, bert_model, tokenizer, BERT_MAX_LEN)
            # )
            # print(f"Explanation: {explanation}")
        else:
            label, confidence = predict(input.text, lstm_model, vocab)
            # explanation = explain_prediction(
            #     input.text,
            #     lambda texts: predict_proba(texts, lstm_model, vocab)
            # )

        logger.info(f"Response | model={input.model} | label={label} | confidence={confidence:.4f}")

        return {
            "text":        input.text,
            "model":       input.model,
            "label":       label,
            "confidence":  confidence,
            #"explanation": explanation
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))