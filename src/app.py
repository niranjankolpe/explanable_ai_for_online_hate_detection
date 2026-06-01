"""
app.py
Streamlit application — Explainable AI for Offensive Language Detection.

Run: streamlit run src/app.py
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yaml
import joblib

from predict       import load_model, predict_proba, get_label_conf
from explain       import explain_prediction
from monitor       import log_prediction, load_logs, compute_drift, get_model_breakdown
from bias_analysis import run_bias_analysis
from preprocess    import preprocess_lstm

with open("params.yaml") as f:
    params = yaml.safe_load(f)


@st.cache_resource
def load_all_models() -> dict:
    models = {}
    for subtask in ["a", "b", "c"]:
        try:
            models[subtask] = {
                m: load_model(m, subtask) for m in ["baseline", "lstm", "bert"]
            }
        except Exception as e:
            models[subtask] = None
            st.warning(f"Could not load subtask {subtask} models: {e}")
    return models


all_models = load_all_models()

st.title("Explainable AI for Offensive Language Detection")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explanation", "Bias Analysis", "Monitoring"])


# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Classify Text")
    text         = st.text_area("Enter text to classify", key="pred_text")
    model_choice = st.selectbox("Select Model", ["baseline", "lstm", "bert"])

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        elif all_models.get("a") is None:
            st.error("Subtask A models not loaded. Train them first.")
        else:
            def run_subtask(subtask):
                model, aux  = all_models[subtask][model_choice]
                proba       = predict_proba([text], model_choice, model, aux)
                label, conf = get_label_conf(proba[0], subtask)
                log_prediction(text, f"{model_choice}_{subtask}", label, conf)
                return label, conf

            label_a, conf_a = run_subtask("a")
            color_a = "red" if label_a == "OFF" else "green"
            st.markdown("### Subtask A — Offensive Language Detection")
            st.markdown(f"**Label:** :{color_a}[{label_a}]  &nbsp;&nbsp; **Confidence:** {conf_a:.4f}")

            if label_a == "NOT":
                st.success("Text is not offensive.")

            elif label_a == "OFF" and all_models.get("b"):
                label_b, conf_b = run_subtask("b")
                type_map = {"TIN": "Targeted Insult/Threat", "UNT": "Untargeted Profanity"}
                color_b  = "red" if label_b == "TIN" else "orange"
                st.markdown("### Subtask B — Offense Type")
                st.markdown(f"**Label:** :{color_b}[{label_b}] — {type_map.get(label_b, label_b)}  "
                            f"&nbsp;&nbsp; **Confidence:** {conf_b:.4f}")

                if label_b == "TIN" and all_models.get("c"):
                    label_c, conf_c = run_subtask("c")
                    target_map = {"IND": "Individual", "GRP": "Group", "OTH": "Other"}
                    st.markdown("### Subtask C — Target Identification")
                    st.markdown(f"**Label:** :red[{label_c}] — {target_map.get(label_c, label_c)}  "
                                f"&nbsp;&nbsp; **Confidence:** {conf_c:.4f}")


# ── Tab 2: Explanation ────────────────────────────────────────────────────────
with tab2:
    st.subheader("LIME + SHAP Explanation")
    st.markdown("Explains which words drive the **Offensive (OFF) vs Not Offensive (NOT)** classification.")
    exp_text = st.text_area("Enter text to explain", key="exp_text")

    if st.button("Explain"):
        if not exp_text.strip():
            st.warning("Please enter some text.")
        elif all_models.get("a") is None:
            st.error("Subtask A LSTM model not loaded.")
        else:
            model, vocab = all_models["a"]["lstm"]
            class_names  = params["subtasks"]["a"]["labels"]  # FIX: dynamic class names

            def lstm_proba(texts):
                return predict_proba(texts, "lstm", model, vocab)

            with st.spinner("Generating explanation (this may take a moment)..."):
                explanation = explain_prediction(exp_text, lstm_proba, class_names)

            words = preprocess_lstm(exp_text).split()

            def plot_bar(scores_dict, title, xlabel):
                df    = pd.DataFrame(scores_dict.items(), columns=["Word", "Score"]).sort_values("Score", ascending=False)
                fig, ax = plt.subplots(figsize=(8, max(1.5, len(df) * 0.65)))
                colors  = ["#DC2626" if s > 0 else "#16A34A" for s in df["Score"]]
                bars    = ax.barh(df["Word"], df["Score"], color=colors)
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_xlabel(xlabel)
                ax.set_title(title)
                xrange = max(df["Score"].max(), 0) - min(df["Score"].min(), 0)
                pad    = xrange * 0.35 if xrange > 0 else 0.1
                ax.set_xlim(min(df["Score"].min(), 0) - pad, max(df["Score"].max(), 0) + pad)
                for bar, val in zip(bars, df["Score"]):
                    ax.text(val + (pad * 0.1 if val >= 0 else -pad * 0.1),
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}", va="center",
                            ha="left" if val >= 0 else "right", fontsize=9)
                plt.tight_layout()
                return fig, df

            def word_highlight(words, scores_dict):
                max_val = max(abs(v) for v in scores_dict.values()) or 1
                html = ""
                for w in words:
                    sc    = scores_dict.get(w, 0)
                    alpha = min(abs(sc) / max_val, 1.0)
                    bg    = f"rgba(220,38,38,{alpha:.2f})" if sc > 0 else f"rgba(22,163,74,{alpha:.2f})"
                    html += f'<span style="background:{bg};padding:3px 6px;margin:2px;border-radius:4px;font-size:16px;">{w}</span> '
                return html

            # LIME
            st.subheader("LIME Explanation")
            fig1, lime_df = plot_bar(explanation["lime"], "LIME Word Importance",
                                     "Score  (positive → OFF,  negative → NOT)")
            st.pyplot(fig1); plt.close(fig1)
            st.markdown(word_highlight(words, explanation["lime"]), unsafe_allow_html=True)
            st.caption("🔴 Red = pushes toward OFF   |   🟢 Green = pushes toward NOT")
            st.dataframe(lime_df)
            st.markdown("---")

            # SHAP
            st.subheader("SHAP Explanation")
            fig2, shap_df = plot_bar(explanation["shap"], "SHAP Word Importance (Shapley Values)",
                                     "SHAP Value  (positive → OFF,  negative → NOT)")
            st.pyplot(fig2); plt.close(fig2)
            st.markdown(word_highlight(words, explanation["shap"]), unsafe_allow_html=True)
            st.caption("🔴 Red = pushes toward OFF   |   🟢 Green = pushes toward NOT")
            st.dataframe(shap_df)
            st.markdown("---")

            # LIME vs SHAP comparison
            st.subheader("LIME vs SHAP — Side-by-Side")
            common = sorted(set(explanation["lime"]) & set(explanation["shap"]))
            comp_df = pd.DataFrame({
                "Word": common,
                "LIME": [explanation["lime"].get(w, 0) for w in common],
                "SHAP": [explanation["shap"].get(w, 0) for w in common],
            }).sort_values("LIME", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(8, max(2, len(comp_df) * 0.65)))
            x, w = range(len(comp_df)), 0.35
            ax3.bar([i - w/2 for i in x], comp_df["LIME"], w, label="LIME", color="#0D9488")
            ax3.bar([i + w/2 for i in x], comp_df["SHAP"], w, label="SHAP", color="#7C3AED")
            ax3.set_xticks(list(x))
            ax3.set_xticklabels(comp_df["Word"], rotation=0, ha="center", fontsize=11)
            ax3.axhline(y=0, color="black", linewidth=0.8)
            ax3.set_ylabel("Score")
            ax3.set_title("LIME vs SHAP — Word Score Comparison")
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3); plt.close(fig3)
            st.caption("LIME: local linear approximation. SHAP: Shapley values from cooperative game theory.")


# ── Tab 3: Bias Analysis ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Responsible AI — Bias Analysis")
    st.markdown(
        "Tests all 3 models on neutral sentences containing identity terms "
        "(race, religion, gender, sexual orientation). "
        "Flagging neutral text as offensive indicates **demographic bias**."
    )

    if st.button("Run Bias Analysis"):
        with st.spinner("Running..."):
            report = run_bias_analysis()
    elif os.path.exists("reports/bias_report.json"):
        st.info("Showing last saved report. Click 'Run Bias Analysis' to refresh.")
        with open("reports/bias_report.json") as f:
            report = json.load(f)
    else:
        report = None

    if report:
        summary    = report["summary"]
        summary_df = pd.DataFrame([
            {"Model": m, "Biased Predictions": summary["bias_counts"][m],
             "Total": summary["total_sentences"], "Bias Rate (%)": summary["bias_rate_percent"][m]}
            for m in ["baseline", "lstm", "bert"]
        ])
        st.dataframe(summary_df)

        st.subheader("Details by Category")
        for category, entries in report["details"].items():
            with st.expander(category.replace("_", " ").title()):
                st.dataframe(pd.DataFrame([{
                    "Sentence":   e["sentence"],
                    "Baseline":   e["baseline"]["label"],
                    "BiLSTM":     e["lstm"]["label"],
                    "DistilBERT": e["bert"]["label"],
                } for e in entries]))


# ── Tab 4: Monitoring ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("Prediction Monitoring & Drift Detection")
    records = load_logs()
    drift   = compute_drift(records)

    if not records:
        st.info("No predictions logged yet.")
    else:
        if drift["drift_detected"]:
            st.error(f"Drift Detected! Offensive rate in last {drift['recent_window']} "
                     f"predictions: {drift['recent_off_rate']*100:.1f}% "
                     f"(threshold: {drift['drift_threshold']*100:.0f}%)")
        else:
            st.success(f"No drift. Offensive rate in last {drift['recent_window']} "
                       f"predictions: {drift['recent_off_rate']*100:.1f}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", drift["total_predictions"])
        c2.metric("Recent OFF Rate",   f"{drift['recent_off_rate']*100:.1f}%")
        c3.metric("Drift Threshold",   f"{drift['drift_threshold']*100:.0f}%")

        st.subheader("By Model")
        bd = get_model_breakdown(records)
        st.dataframe(pd.DataFrame([
            {"Model": m, "Total": v["total"],
             "Labels": "  |  ".join(f"{k}: {c}" for k, c in v.items() if k != "total")}
            for m, v in bd.items()
        ]))

        st.subheader("Recent Predictions")
        st.dataframe(pd.DataFrame(records[-20:][::-1]))

        if st.button("Clear Logs"):
            os.remove(LOG_FILE) if os.path.exists(LOG_FILE) else None
            st.success("Logs cleared.")
            st.rerun()
