import streamlit as st
import yaml
import joblib

from predict import load_model, predict, predict_proba
from predict_bert import load_bert_model, predict_bert, predict_bert_proba
from explain import explain_prediction

with open("params.yaml") as f:
    params = yaml.safe_load(f)

BERT_MAX_LEN = params["bert"]["max_len"]

baseline_model = joblib.load("models/baseline/baseline_model.pkl")
vectorizer     = joblib.load("models/baseline/tfidf_vectorizer.pkl")


@st.cache_resource
def load_all():
    lstm_model, vocab         = load_model()
    bert_model, tokenizer     = load_bert_model()
    return lstm_model, vocab, bert_model, tokenizer


lstm_model, vocab, bert_model, tokenizer = load_all()

st.title("Offensive Language Detection")
st.markdown("Explainable AI Framework for Online Hate Detection — OLID Dataset")

text         = st.text_area("Enter text to classify")
model_choice = st.selectbox("Select Model", ["baseline", "lstm", "bert"])
show_explain = st.checkbox("Show Explanation (LIME + SHAP)", value=True)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice == "baseline":
            X     = vectorizer.transform([text.lower()])
            label = baseline_model.predict(X)[0]
            confidence = "N/A"
            st.subheader("Result")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {confidence}")
            st.info("Explanation not available for baseline model.")

        elif model_choice == "bert":
            label, confidence = predict_bert(text, bert_model, tokenizer, BERT_MAX_LEN)
            st.subheader("Result")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {confidence:.4f}")
            if show_explain:
                with st.spinner("Generating explanation..."):
                    explanation = explain_prediction(
                        text,
                        lambda texts: predict_bert_proba(texts, bert_model, tokenizer, BERT_MAX_LEN)
                    )
                st.subheader("Explanation")
                st.write("**LIME:**", explanation["lime"])
                st.write("**SHAP:**", explanation["shap"])

        else:  # lstm
            label, confidence = predict(text, lstm_model, vocab)
            st.subheader("Result")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {confidence:.4f}")
            if show_explain:
                with st.spinner("Generating explanation..."):
                    explanation = explain_prediction(
                        text,
                        lambda texts: predict_proba(texts, lstm_model, vocab)
                    )
                st.subheader("Explanation")
                st.write("**LIME:**", explanation["lime"])
                st.write("**SHAP:**", explanation["shap"])