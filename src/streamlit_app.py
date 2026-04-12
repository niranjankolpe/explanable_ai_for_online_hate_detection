import streamlit as st
import yaml
import joblib

from predict import load_model, predict
from predict_bert import load_bert_model, predict_bert
from preprocessing import preprocess

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

BERT_MAX_LEN = params["bert"]["max_len"]

# Load baseline
baseline_model = joblib.load("models/baseline/baseline_model.pkl")
vectorizer     = joblib.load("models/baseline/tfidf_vectorizer.pkl")

# Load models
@st.cache_resource
def load_all():
    lstm_model, vocab = load_model()
    bert_model, tokenizer = load_bert_model()
    return lstm_model, vocab, bert_model, tokenizer

lstm_model, vocab, bert_model, tokenizer = load_all()

st.title("Offensive Language Detection")

text = st.text_area("Enter text")

model_choice = st.selectbox(
    "Select Model",
    ["baseline", "lstm", "bert"]
)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        #print(f"Original Text: {text}")
        text = preprocess(text)
        #print(f"Preprocessed Text: {text}")
        if model_choice == "baseline":
            X = vectorizer.transform([text.lower()])
            pred = baseline_model.predict(X)[0]
            label = pred
            confidence = "N/A"

        elif model_choice == "bert":
            label, confidence = predict_bert(
                text, bert_model, tokenizer, BERT_MAX_LEN
            )

        else:
            label, confidence = predict(
                text, lstm_model, vocab
            )

        st.subheader("Result")
        st.write(f"Label: {label}")
        st.write(f"Confidence: {confidence}")