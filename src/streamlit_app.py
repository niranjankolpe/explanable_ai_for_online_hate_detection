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

BERT_MAX_LEN = params["bert"]["max_len"]


@st.cache_resource
def load_all_models():
    models = {}
    for subtask in ["a", "b", "c"]:
        try:
            lstm_model, vocab     = load_model(subtask)
            bert_model, tokenizer = load_bert_model(subtask)
            baseline_model        = joblib.load(f"models/baseline_{subtask}/baseline_model.pkl")
            vectorizer            = joblib.load(f"models/baseline_{subtask}/tfidf_vectorizer.pkl")
            models[subtask] = {
                "lstm":      (lstm_model, vocab),
                "bert":      (bert_model, tokenizer),
                "baseline":  (baseline_model, vectorizer)
            }
        except Exception as e:
            models[subtask] = None
            print(f"Could not load subtask {subtask} models: {e}")
    return models


all_models = load_all_models()

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
        elif all_models.get("a") is None:
            st.error("Subtask A model not loaded. Train it first.")
        else:
            # ── Step 1: Subtask A ──────────────────────────────────────────
            if model_choice == "baseline":
                baseline_a, vec_a = all_models["a"]["baseline"]
                X_a       = vec_a.transform([text.lower()])
                label_a   = baseline_a.predict(X_a)[0]
                proba_a   = baseline_a.predict_proba(X_a)[0]
                conf_a    = float(max(proba_a))
            elif model_choice == "bert":
                bert_a, tok_a   = all_models["a"]["bert"]
                label_a, conf_a = predict_bert(text, bert_a, tok_a, BERT_MAX_LEN, "a")
            else:
                lstm_a, vocab_a = all_models["a"]["lstm"]
                label_a, conf_a = predict(text, lstm_a, vocab_a, "a")

            log_prediction(text, f"{model_choice}_a", label_a, conf_a)

            color_a = "red" if label_a == "OFF" else "green"
            st.markdown("### Subtask A — Offensive Language Detection")
            st.markdown(f"**Label:** :{color_a}[{label_a}] &nbsp;&nbsp; **Confidence:** {conf_a:.4f}")

            if label_a == "NOT":
                st.success("Text is not offensive. No further analysis needed.")

            # ── Step 2: Subtask B (only if OFF) ───────────────────────────
            elif label_a == "OFF":
                if all_models.get("b") is None:
                    st.warning("Subtask B model not loaded.")
                else:
                    if model_choice == "baseline":
                        baseline_b, vec_b = all_models["b"]["baseline"]
                        X_b       = vec_b.transform([text.lower()])
                        label_b   = baseline_b.predict(X_b)[0]
                        proba_b   = baseline_b.predict_proba(X_b)[0]
                        conf_b    = float(max(proba_b))
                    elif model_choice == "bert":
                        bert_b, tok_b   = all_models["b"]["bert"]
                        label_b, conf_b = predict_bert(text, bert_b, tok_b, BERT_MAX_LEN, "b")
                    else:
                        lstm_b, vocab_b = all_models["b"]["lstm"]
                        label_b, conf_b = predict(text, lstm_b, vocab_b, "b")

                    log_prediction(text, f"{model_choice}_b", label_b, conf_b)

                    type_map = {"TIN": "Targeted Insult/Threat", "UNT": "Untargeted Profanity"}
                    color_b  = "red" if label_b == "TIN" else "orange"
                    st.markdown("### Subtask B — Offense Type")
                    st.markdown(f"**Label:** :{color_b}[{label_b}] — {type_map.get(label_b, label_b)} &nbsp;&nbsp; **Confidence:** {conf_b:.4f}")

                    if label_b == "UNT":
                        st.info("Offense is untargeted (general profanity). No target identification needed.")

                    # ── Step 3: Subtask C (only if TIN) ───────────────────
                    elif label_b == "TIN":
                        if all_models.get("c") is None:
                            st.warning("Subtask C model not loaded.")
                        else:
                            if model_choice == "baseline":
                                baseline_c, vec_c = all_models["c"]["baseline"]
                                X_c       = vec_c.transform([text.lower()])
                                label_c   = baseline_c.predict(X_c)[0]
                                proba_c   = baseline_c.predict_proba(X_c)[0]
                                conf_c    = float(max(proba_c))
                            elif model_choice == "bert":
                                bert_c, tok_c   = all_models["c"]["bert"]
                                label_c, conf_c = predict_bert(text, bert_c, tok_c, BERT_MAX_LEN, "c")
                            else:
                                lstm_c, vocab_c = all_models["c"]["lstm"]
                                label_c, conf_c = predict(text, lstm_c, vocab_c, "c")

                            log_prediction(text, f"{model_choice}_c", label_c, conf_c)

                            target_map = {"IND": "Individual", "GRP": "Group", "OTH": "Other"}
                            color_c    = "red"
                            st.markdown("### Subtask C — Target Identification")
                            st.markdown(f"**Label:** :{color_c}[{label_c}] — {target_map.get(label_c, label_c)} &nbsp;&nbsp; **Confidence:** {conf_c:.4f}")
# ── Tab 2: Explanation ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("LIME + SHAP Explanation (LSTM only)")
    exp_text    = st.text_area("Enter text to explain", key="exp_text")
    exp_subtask = st.selectbox("Subtask", ["a", "b", "c"], key="exp_subtask")

    if st.button("Explain"):
        if not exp_text.strip():
            st.warning("Please enter some text.")
        elif all_models.get(exp_subtask) is None:
            st.error(f"Subtask {exp_subtask.upper()} models not loaded.")
        else:
            lstm_model, vocab = all_models[exp_subtask]["lstm"]
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
        "Tracks all predictions made via the app. "
        "Drift is flagged if the offensive rate exceeds 60% "
        "in the last 20 predictions."
    )

    records = load_logs()
    drift   = compute_drift(records)

    if not records:
        st.info("No predictions logged yet. Make some predictions in the Prediction tab.")
    else:
        if drift["drift_detected"]:
            st.error(
                f"Drift Detected! Offensive rate in last "
                f"{drift['recent_window']} predictions: "
                f"{drift['recent_off_rate']*100:.1f}% "
                f"(threshold: {drift['drift_threshold']*100:.0f}%)"
            )
        else:
            st.success(
                f"No drift detected. Offensive rate in last "
                f"{drift['recent_window']} predictions: "
                f"{drift['recent_off_rate']*100:.1f}%"
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", drift["total_predictions"])
        col2.metric("Recent OFF Rate",   f"{drift['recent_off_rate']*100:.1f}%")
        col3.metric("Drift Threshold",   f"{drift['drift_threshold']*100:.0f}%")

        st.subheader("Predictions by Model")
        breakdown = get_model_breakdown(records)
        bd_rows = []
        for m, counts in breakdown.items():
            total = counts["total"]
            labels = {k: v for k, v in counts.items() if k != "total"}
            label_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
            bd_rows.append({
                "Model":    m,
                "Total":    total,
                "Labels":   label_str,
            })
        st.dataframe(pd.DataFrame(bd_rows))

        st.subheader("Recent Predictions")
        recent_df = pd.DataFrame(records[-20:][::-1])
        st.dataframe(recent_df)

        if st.button("Clear Logs"):
            os.remove("logs/predictions.log")
            st.success("Logs cleared.")
            st.rerun()