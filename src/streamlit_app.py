import os
import json
import streamlit as st
import yaml
import joblib
import pandas as pd

from predict import load_model, predict, predict_proba
from predict_bert import load_bert_model, predict_bert, predict_bert_proba
from explain import explain_prediction
from monitor import log_prediction, load_logs, compute_drift, get_model_breakdown

with open("params.yaml") as f:
    params = yaml.safe_load(f)

BERT_MAX_LEN   = params["bert"]["max_len"]
baseline_model = joblib.load("models/baseline/baseline_model.pkl")
vectorizer     = joblib.load("models/baseline/tfidf_vectorizer.pkl")


@st.cache_resource
def load_all():
    lstm_model, vocab     = load_model()
    bert_model, tokenizer = load_bert_model()
    return lstm_model, vocab, bert_model, tokenizer


lstm_model, vocab, bert_model, tokenizer = load_all()

st.title("Explainable AI for Offensive Language Detection")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explanation", "Bias Analysis", "Monitoring"])

# ── Tab 1: Prediction ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Classify Text")
    text         = st.text_area("Enter text to classify", key="pred_text")
    model_choice = st.selectbox("Select Model", ["baseline", "lstm", "bert"])

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            if model_choice == "baseline":
                X     = vectorizer.transform([text.lower()])
                label = baseline_model.predict(X)[0]
                proba = baseline_model.predict_proba(X)[0]
                confidence = float(max(proba))
                log_prediction(text, "baseline", label, confidence)
                color = "red" if label == "OFF" else "green"
                st.markdown(f"**Label:** :{color}[{label}]")
                st.write(f"**Confidence:** {confidence:.4f}")

            elif model_choice == "bert":
                label, confidence = predict_bert(text, bert_model, tokenizer, BERT_MAX_LEN)
                log_prediction(text, "bert", label, confidence)
                color = "red" if label == "OFF" else "green"
                st.markdown(f"**Label:** :{color}[{label}]")
                st.write(f"**Confidence:** {confidence:.4f}")

            else:
                label, confidence = predict(text, lstm_model, vocab)
                log_prediction(text, "lstm", label, confidence)
                color = "red" if label == "OFF" else "green"
                st.markdown(f"**Label:** :{color}[{label}]")
                st.write(f"**Confidence:** {confidence:.4f}")

# ── Tab 2: Explanation ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("LIME + SHAP Explanation (LSTM only)")
    exp_text = st.text_area("Enter text to explain", key="exp_text")

    if st.button("Explain"):
        if not exp_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating explanation (this may take a minute)..."):
                explanation = explain_prediction(
                    exp_text,
                    lambda texts: predict_proba(texts, lstm_model, vocab)
                )

            st.subheader("LIME Explanation")
            lime_df = pd.DataFrame(
                explanation["lime"].items(),
                columns=["Word", "Score"]
            ).sort_values("Score", ascending=False)
            st.dataframe(lime_df)

            st.subheader("SHAP Explanation")
            shap_df = pd.DataFrame(
                explanation["shap"].items(),
                columns=["Word", "Score"]
            ).sort_values("Score", ascending=False)
            st.dataframe(shap_df)

# ── Tab 3: Bias Analysis ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Responsible AI — Bias Analysis")
    st.markdown(
        "Tests all 3 models on neutral sentences containing identity terms "
        "(race, religion, gender, sexual orientation). "
        "A model flagging these as offensive indicates **demographic bias**."
    )

    if st.button("Run Bias Analysis"):
        with st.spinner("Running bias analysis across all models..."):
            from bias_analysis import run_bias_analysis
            report = run_bias_analysis()

        summary = report["summary"]
        st.subheader("Summary")
        summary_df = pd.DataFrame([
            {
                "Model":              m,
                "Biased Predictions": summary["bias_counts"][m],
                "Total Sentences":    summary["total_sentences"],
                "Bias Rate (%)":      summary["bias_rate_percent"][m]
            }
            for m in ["baseline", "lstm", "bert"]
        ])
        st.dataframe(summary_df)

        st.subheader("Detailed Results by Category")
        for category, entries in report["details"].items():
            with st.expander(f"{category.replace('_', ' ').title()}"):
                rows = []
                for e in entries:
                    rows.append({
                        "Sentence":   e["sentence"],
                        "Baseline":   e["baseline"]["label"],
                        "BiLSTM":     e["lstm"]["label"],
                        "DistilBERT": e["bert"]["label"],
                    })
                st.dataframe(pd.DataFrame(rows))

    elif os.path.exists("reports/bias_report.json"):
        st.info("Showing last saved bias report. Click 'Run Bias Analysis' to refresh.")
        with open("reports/bias_report.json") as f:
            report = json.load(f)
        summary = report["summary"]
        summary_df = pd.DataFrame([
            {
                "Model":              m,
                "Biased Predictions": summary["bias_counts"][m],
                "Total Sentences":    summary["total_sentences"],
                "Bias Rate (%)":      summary["bias_rate_percent"][m]
            }
            for m in ["baseline", "lstm", "bert"]
        ])
        st.dataframe(summary_df)

# ── Tab 4: Monitoring ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Prediction Monitoring & Drift Detection")
    st.markdown(
        f"Tracks all predictions made via the app. "
        f"Drift is flagged if the offensive rate exceeds **60%** "
        f"in the last **20 predictions**."
    )

    records = load_logs()
    drift   = compute_drift(records)

    if not records:
        st.info("No predictions logged yet. Make some predictions in the Prediction tab.")
    else:
        # Drift alert
        if drift["drift_detected"]:
            st.error(
                f"⚠️ Drift Detected! Offensive rate in last "
                f"{drift['recent_window']} predictions: "
                f"{drift['recent_off_rate']*100:.1f}% "
                f"(threshold: {drift['drift_threshold']*100:.0f}%)"
            )
        else:
            st.success(
                f"✅ No drift detected. Offensive rate in last "
                f"{drift['recent_window']} predictions: "
                f"{drift['recent_off_rate']*100:.1f}%"
            )

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", drift["total_predictions"])
        col2.metric("Recent OFF Rate",   f"{drift['recent_off_rate']*100:.1f}%")
        col3.metric("Drift Threshold",   f"{drift['drift_threshold']*100:.0f}%")

        # Model breakdown
        st.subheader("Predictions by Model")
        breakdown = get_model_breakdown(records)
        bd_rows   = []
        for m, counts in breakdown.items():
            off_rate = counts["OFF"] / counts["total"] * 100 if counts["total"] > 0 else 0
            bd_rows.append({
                "Model":       m,
                "Total":       counts["total"],
                "OFF":         counts["OFF"],
                "NOT":         counts["NOT"],
                "OFF Rate (%)": round(off_rate, 1)
            })
        st.dataframe(pd.DataFrame(bd_rows))

        # Recent predictions table
        st.subheader("Recent Predictions")
        recent_df = pd.DataFrame(records[-20:][::-1])
        st.dataframe(recent_df)

        if st.button("Clear Logs"):
            os.remove("logs/predictions.log")
            st.success("Logs cleared.")
            st.rerun()