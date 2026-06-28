"""
app.py
Streamlit application — Explainable AI for Offensive Language Detection.

Run: streamlit run src/app.py
"""

from rag_engine import load_vector_store, retrieve_similar, generate_explanation
from crawler import scrape_multiple, save_crawled_data
from preprocess import preprocess_common
from bias_analysis import run_bias_analysis
from monitor import log_prediction, load_logs, compute_drift, get_model_breakdown
from explain import explain_prediction
from predict import load_model, predict_proba, get_label_conf
import joblib
import yaml
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import json
import asyncio
import websockets

import matplotlib
matplotlib.use("Agg")


with open("params.yaml") as f:
    params = yaml.safe_load(f)

# --- LAZY LOADING MEMORY OPTIMIZATION ---
# By setting max_entries=2, Streamlit will only ever keep 2 models in RAM at a time.
# If you load a 3rd model, it deletes the oldest one from RAM. This
# prevents 8GB laptops from crashing!


@st.cache_resource(max_entries=2)
def get_model_cached(model_type: str, subtask: str):
    if model_type == "llama" and subtask != "a":
        return None  # Llama adapter was only trained for subtask A

    try:
        return load_model(model_type, subtask)
    except Exception as e:
        st.error(f"Could not load {model_type} for subtask {subtask}: {e}")
        return None


# Dynamically check which models are available on disk without loading
# them into memory
available_models = ["baseline", "lstm", "bert"]
if os.path.exists("models/llama3.2_3b_lora_hate_speech"):
    available_models.append("llama")

st.title("Explainable AI for Offensive Language Detection")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Prediction", "Explanation", "RAG Explainer", "Bias Analysis", "Data Collection", "Monitoring", "Live Stream Moderation", "Agentic Analysis"]
)


# ── Tab 1: Prediction ───────────────────────────────────────────────────
with tab1:
    st.subheader("Classify Text")
    text = st.text_area("Enter text to classify", key="pred_text")
    model_choice = st.selectbox("Select Model", available_models)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            def run_subtask(subtask):
                cached = get_model_cached(model_choice, subtask)
                if cached is None:
                    return None, None
                model, aux = cached
                proba = predict_proba(
                    [text], model_choice, model, aux, subtask)
                label, conf = get_label_conf(proba[0], subtask)
                log_prediction(text, f"{model_choice}_{subtask}", label, conf)
                return label, conf

            with st.spinner(f"Loading {model_choice} (this may take a bit for Llama)..."):
                label_a, conf_a = run_subtask("a")

            if label_a is not None:
                color_a = "red" if label_a == "OFF" else "green"
                st.markdown("### Subtask A — Offensive Language Detection")
                st.markdown(
                    f"**Label:** :{color_a}[{label_a}]  &nbsp;&nbsp; **Confidence:** {conf_a:.4f}")

                if label_a == "NOT":
                    st.success("Text is not offensive.")

                elif label_a == "OFF" and model_choice != "llama":
                    label_b, conf_b = run_subtask("b")
                    if label_b is not None:
                        type_map = {
                            "TIN": "Targeted Insult/Threat",
                            "UNT": "Untargeted Profanity"}
                        color_b = "red" if label_b == "TIN" else "orange"
                        st.markdown("### Subtask B — Offense Type")
                        st.markdown(
                            f"**Label:** :{color_b}[{label_b}] — {type_map.get(label_b, label_b)}  "
                            f"&nbsp;&nbsp; **Confidence:** {conf_b:.4f}")

                        if label_b == "TIN":
                            label_c, conf_c = run_subtask("c")
                            if label_c is not None:
                                target_map = {
                                    "IND": "Individual", "GRP": "Group", "OTH": "Other"}
                                st.markdown(
                                    "### Subtask C — Target Identification")
                                st.markdown(
                                    f"**Label:** :red[{label_c}] — {target_map.get(label_c, label_c)}  "
                                    f"&nbsp;&nbsp; **Confidence:** {conf_c:.4f}")


# ── Tab 2: Explanation ──────────────────────────────────────────────────
with tab2:
    st.subheader("LIME + SHAP Explanation")
    st.markdown(
        "Explains which words drive the **Subtask A (OFF vs NOT)** classification for the selected model.")
    if "llama" in available_models:
        st.warning("⚠️ **Warning:** The Llama model has 3 Billion parameters. Running LIME/SHAP on it using a laptop CPU will take **30+ minutes** per sentence! Use with caution.")
    exp_text = st.text_area("Enter text to explain", key="exp_text")
    exp_model = st.selectbox(
        "Select Model to Explain",
        available_models,
        key="exp_model")

    if st.button("Explain"):
        if not exp_text.strip():
            st.warning("Please enter some text.")
        else:
            cached = get_model_cached(exp_model, "a")
            if cached is None:
                st.error(f"Subtask A {exp_model} model failed to load.")
            else:
                model, aux = cached
                class_names = params["subtasks"]["a"]["labels"]

                def explain_proba(texts):
                    return predict_proba(texts, exp_model, model, aux, "a")

                with st.spinner("Generating explanation (this may take a moment)..."):
                    explanation = explain_prediction(
                        exp_text, explain_proba, class_names)

                words = preprocess_common(exp_text).split()

                def plot_bar(scores_dict, title, xlabel):
                    df = pd.DataFrame(
                        scores_dict.items(), columns=[
                            "Word", "Score"]).sort_values(
                        "Score", ascending=False)
                    fig, ax = plt.subplots(
                        figsize=(8, max(1.5, len(df) * 0.65)))
                    colors = [
                        "#DC2626" if s > 0 else "#16A34A" for s in df["Score"]]
                    bars = ax.barh(df["Word"], df["Score"], color=colors)
                    ax.axvline(x=0, color="black", linewidth=0.8)
                    ax.set_xlabel(xlabel)
                    ax.set_title(title)
                    xrange = max(df["Score"].max(), 0) - \
                        min(df["Score"].min(), 0)
                    pad = xrange * 0.35 if xrange > 0 else 0.1
                    ax.set_xlim(min(df["Score"].min(), 0) -
                                pad, max(df["Score"].max(), 0) + pad)
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
                        sc = scores_dict.get(w, 0)
                        alpha = min(abs(sc) / max_val, 1.0)
                        bg = f"rgba(220,38,38,{alpha:.2f})" if sc > 0 else f"rgba(22,163,74,{alpha:.2f})"
                        html += f'<span style="background:{bg};padding:3px 6px;margin:2px;border-radius:4px;font-size:16px;">{w}</span> '
                    return html

                # LIME
                st.subheader("LIME Explanation")
                fig1, lime_df = plot_bar(explanation["lime"], "LIME Word Importance",
                                         "Score  (positive → OFF,  negative → NOT)")
                st.pyplot(fig1)
                plt.close(fig1)
                st.markdown(
                    word_highlight(
                        words,
                        explanation["lime"]),
                    unsafe_allow_html=True)
                st.caption(
                    "🔴 Red = pushes toward OFF   |   🟢 Green = pushes toward NOT")
                st.dataframe(lime_df)
                st.markdown("---")

                # SHAP
                st.subheader("SHAP Explanation")
                fig2, shap_df = plot_bar(explanation["shap"], "SHAP Word Importance (Shapley Values)",
                                         "SHAP Value  (positive → OFF,  negative → NOT)")
                st.pyplot(fig2)
                plt.close(fig2)
                st.markdown(
                    word_highlight(
                        words,
                        explanation["shap"]),
                    unsafe_allow_html=True)
                st.caption(
                    "🔴 Red = pushes toward OFF   |   🟢 Green = pushes toward NOT")
                st.dataframe(shap_df)
                st.markdown("---")

                # LIME vs SHAP comparison
                st.subheader("LIME vs SHAP — Side-by-Side")
                common = sorted(
                    set(explanation["lime"]) & set(explanation["shap"]))
                comp_df = pd.DataFrame({
                    "Word": common,
                    "LIME": [explanation["lime"].get(w, 0) for w in common],
                    "SHAP": [explanation["shap"].get(w, 0) for w in common],
                }).sort_values("LIME", ascending=False)

                fig3, ax3 = plt.subplots(
                    figsize=(8, max(2, len(comp_df) * 0.65)))
                x, w = range(len(comp_df)), 0.35
                ax3.bar([i - w / 2 for i in x], comp_df["LIME"],
                        w, label="LIME", color="#0D9488")
                ax3.bar([i + w / 2 for i in x], comp_df["SHAP"],
                        w, label="SHAP", color="#7C3AED")
                ax3.set_xticks(list(x))
                ax3.set_xticklabels(
                    comp_df["Word"],
                    rotation=0,
                    ha="center",
                    fontsize=11)
                ax3.axhline(y=0, color="black", linewidth=0.8)
                ax3.set_ylabel("Score")
                ax3.set_title("LIME vs SHAP — Word Score Comparison")
                ax3.legend()
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
                st.caption(
                    "LIME: local linear approximation. SHAP: Shapley values from cooperative game theory.")


# ── Tab 4: Bias Analysis ────────────────────────────────────────────────
with tab4:
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
        summary = report["summary"]
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
                    "Sentence": e["sentence"],
                    "Baseline": e["baseline"]["label"],
                    "BiLSTM": e["lstm"]["label"],
                    "DistilBERT": e["bert"]["label"],
                } for e in entries]))


# ── Tab 6: Monitoring ───────────────────────────────────────────────────
with tab6:
    st.subheader("Prediction Monitoring & Drift Detection")
    records = load_logs()
    drift = compute_drift(records)

    if not records:
        st.info("No predictions logged yet.")
    else:
        if drift["drift_detected"]:
            st.error(
                f"Drift Detected! Offensive rate in last {drift['recent_window']} "
                f"predictions: {drift['recent_off_rate']*100:.1f}% "
                f"(threshold: {drift['drift_threshold']*100:.0f}%)")
        else:
            st.success(
                f"No drift. Offensive rate in last {drift['recent_window']} "
                f"predictions: {drift['recent_off_rate']*100:.1f}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", drift["total_predictions"])
        c2.metric("Recent OFF Rate", f"{drift['recent_off_rate']*100:.1f}%")
        c3.metric("Drift Threshold", f"{drift['drift_threshold']*100:.0f}%")

        st.subheader("By Model")
        bd = get_model_breakdown(records)
        st.dataframe(pd.DataFrame([
            {"Model": m, "Total": v["total"],
             "Labels": "  |  ".join(f"{k}: {c}" for k, c in v.items() if k != "total")}
            for m, v in bd.items()
        ]))

        st.subheader("Recent Predictions")
        st.dataframe(pd.DataFrame(records[-20:][::-1]))

        log_file = "logs/predictions.log"
        if st.button("Clear Logs"):
            os.remove(log_file) if os.path.exists(log_file) else None
            st.success("Logs cleared.")
            st.rerun()


# ── Tab 3: RAG Explainer ────────────────────────────────────────────────
with tab3:
    st.subheader("AI-Powered Explanation (RAG)")
    st.markdown(
        "Uses **Retrieval-Augmented Generation** to explain predictions in plain language. "
        "Retrieves similar tweets from the training data, combines with LIME/SHAP analysis, "
        "and sends to Google Gemini for a human-readable explanation.")

    rag_text = st.text_area("Enter text to explain", key="rag_text")
    rag_model = st.selectbox(
        "Select Classifier",
        available_models,
        key="rag_model")
    rag_llm = st.selectbox(
        "Select Explainer LLM", [
            "Gemini", "Ollama (Llama 3.1)"], key="rag_llm")
    rag_key = st.text_input(
        "Google API Key",
        type="password",
        key="rag_api_key",
        value=os.environ.get(
            "GOOGLE_API_KEY",
            ""),
        help="Required for Gemini. Get a free key from https://aistudio.google.com/apikey")

    # Check if vector store exists
    _chroma_ready = os.path.exists("models/chroma_store")
    if not _chroma_ready:
        st.warning(
            "ChromaDB vector store not built yet. Run: `python src/build_vector_store.py`")

    if st.button("Explain with AI", disabled=not _chroma_ready):
        if not rag_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating explanation..."):
                cached = get_model_cached(rag_model, "a")
                if cached is None:
                    st.error(f"Subtask A {rag_model} model failed to load.")
                else:
                    model, aux = cached
                    proba = predict_proba(
                        [rag_text], rag_model, model, aux, "a")
                    label, conf = get_label_conf(proba[0], "a")

                    # 2. LIME + SHAP
                    class_names = params["subtasks"]["a"]["labels"]

                    def rag_proba(texts):
                        return predict_proba(texts, rag_model, model, aux, "a")
                    explanation = explain_prediction(
                        rag_text, rag_proba, class_names)

                    # 3. Retrieve similar
                    try:
                        collection = load_vector_store()
                        similar = retrieve_similar(rag_text, collection, k=5)
                    except Exception as e:
                        similar = []
                        st.warning(f"Could not retrieve similar tweets: {e}")

                    # 4. Generate LLM explanation
                    llm_explanation = generate_explanation(
                        text=rag_text,
                        model_name=rag_model,
                        prediction=label,
                        confidence=conf,
                        lime_scores=explanation["lime"],
                        shap_scores=explanation["shap"],
                        similar_examples=similar,
                        api_key=rag_key,
                        llm_provider=rag_llm,
                    )

                    # Display results
                    color = "red" if label == "OFF" else "green"
                    st.markdown(
                        f"### Prediction: :{color}[{label}] (confidence: {conf:.4f})")

                    st.markdown("### AI Explanation")
                    st.markdown(llm_explanation)

                    with st.expander("Similar tweets from training data"):
                        if similar:
                            sim_df = pd.DataFrame([{
                                "Tweet": s["tweet"][:100],
                                "Label A": s["label_a"],
                                "Label B": s["label_b"],
                                "Label C": s["label_c"],
                            } for s in similar])
                            st.dataframe(sim_df)
                        else:
                            st.info("No similar tweets found.")

                    with st.expander("LIME + SHAP word scores"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**LIME**")
                            for word, score in sorted(
                                explanation["lime"].items(), key=lambda x: abs(
                                    x[1]), reverse=True):
                                direction = "OFF" if score > 0 else "NOT"
                                st.text(
                                    f"  {word}: {score:+.4f} ({direction})")
                        with col2:
                            st.markdown("**SHAP**")
                            for word, score in sorted(
                                explanation["shap"].items(), key=lambda x: abs(
                                    x[1]), reverse=True):
                                direction = "OFF" if score > 0 else "NOT"
                                st.text(
                                    f"  {word}: {score:+.4f} ({direction})")


# ── Tab 5: Data Collection ──────────────────────────────────────────────
with tab5:
    st.subheader("Web Data Collection")
    st.markdown(
        "Scrape text from web pages and run hate speech detection on the collected content. "
        "Paste one URL per line. Results are saved to CSV.")

    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/article1\nhttps://example.com/article2",
        key="crawl_urls",
        height=120,
    )
    crawl_model = st.selectbox(
        "Select Model for Analysis",
        available_models,
        key="crawl_model")

    # Clear results if model changes
    if "last_crawl_model" in st.session_state and st.session_state[
            "last_crawl_model"] != crawl_model:
        if "crawled_results" in st.session_state:
            del st.session_state["crawled_results"]
    st.session_state["last_crawl_model"] = crawl_model

    if st.button("Scrape & Analyze"):
        urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            # 1. Scrape
            with st.spinner(f"Scraping {len(urls)} URL(s)..."):
                results = scrape_multiple(urls)

            # Collect all text chunks
            all_texts = []
            for url, data in results.items():
                if data["status"] == "ok":
                    for text in data["texts"]:
                        all_texts.append({"url": url, "text": text})

            if not all_texts:
                st.warning("No text content extracted from the URLs.")
                if "crawled_results" in st.session_state:
                    del st.session_state["crawled_results"]
            else:
                # 2. Predict
                with st.spinner(f"Running predictions on {len(all_texts)} text chunks..."):
                    cached = get_model_cached(crawl_model, "a")
                    if cached is None:
                        st.error(f"Failed to load {crawl_model}.")
                    else:
                        model, aux = cached
                        texts_list = [t["text"] for t in all_texts]
                        probas = predict_proba(
                            texts_list, crawl_model, model, aux, "a")

                        pred_rows = []
                        for i, t in enumerate(all_texts):
                            label, conf = get_label_conf(probas[i], "a")
                            pred_rows.append({
                                "Source URL": t["url"],
                                "Text": t["text"][:150],
                                "Prediction": label,
                                "Confidence": round(conf, 4),
                            })

                        pred_df = pd.DataFrame(pred_rows)

                        # Save crawled data locally
                        csv_path = save_crawled_data(results)

                        # Persist in session state
                        st.session_state["crawled_results"] = {
                            "results": results,
                            "pred_df": pred_df,
                            "csv_path": csv_path,
                        }

    # Render results if they exist in session state
    if "crawled_results" in st.session_state:
        cdata = st.session_state["crawled_results"]
        results = cdata["results"]
        pred_df = cdata["pred_df"]
        csv_path = cdata["csv_path"]

        # Show scraping status
        ok_count = sum(1 for v in results.values() if v["status"] == "ok")
        err_count = sum(1 for v in results.values() if v["status"] == "error")
        st.info(f"Scraped {ok_count} URL(s) successfully, {err_count} failed.")

        for url, data in results.items():
            if data["status"] == "error":
                st.warning(f"Failed: {url} — {data['error']}")

        # Summary stats
        total = len(pred_df)
        off_count = (pred_df["Prediction"] == "OFF").sum()
        not_count = (pred_df["Prediction"] == "NOT").sum()
        off_rate = off_count / total * 100 if total > 0 else 0

        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Texts", total)
        c2.metric("Offensive", int(off_count))
        c3.metric("Not Offensive", int(not_count))
        c4.metric("Offensive Rate", f"{off_rate:.1f}%")

        # Results table
        st.markdown("### Detailed Results")
        st.dataframe(pred_df)

        if csv_path:
            st.success(f"Raw crawled data saved to: {csv_path}")

        # Download results
        csv_download = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_download,
            file_name="crawl_predictions.csv",
            mime="text/csv",
        )


# ── Tab 7: Live Stream Moderation ───────────────────────────────────────
with tab7:
    st.subheader("Live Social Media Moderation Stream")
    st.markdown(
        "Connects directly to the **Bluesky Jetstream public firehose** via WebSockets. "
        "Fetches and moderates incoming public posts in real-time.")

    stream_model = st.selectbox(
        "Select Model for Live Analysis",
        available_models,
        key="stream_model")
    max_posts = st.slider(
        "Number of posts to capture",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        key="stream_max_posts")

    async def run_bluesky_stream(model_name, limit):
        cached = get_model_cached(model_name, "a")
        if cached is None:
            st.error("Model failed to load.")
            return
        model, aux = cached

        status_box = st.empty()
        progress_bar = st.progress(0)
        feed_container = st.empty()

        results = []
        url = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

        status_box.info("Connecting to Bluesky firehose...")

        try:
            async with websockets.connect(url) as websocket:
                status_box.success("Connected! Listening for live posts...")

                count = 0
                while count < limit:
                    message = await websocket.recv()
                    data = json.loads(message)

                    post_text = data.get(
                        "commit",
                        {}).get(
                        "record",
                        {}).get(
                        "text",
                        "")
                    langs = data.get(
                        "commit",
                        {}).get(
                        "record",
                        {}).get(
                        "langs",
                        [])
                    # Filter out short/empty/non-english posts
                    if post_text and len(
                            post_text.strip()) > 15 and "en" in langs:
                        # Clean and predict
                        proba = predict_proba(
                            [post_text], model_name, model, aux, "a")
                        label, conf = get_label_conf(proba[0], "a")

                        results.append({
                            "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "text": post_text.strip(),
                            "label": label,
                            "conf": conf
                        })

                        count += 1
                        progress_bar.progress(count / limit)

                        # Render scrolling feed in reverse chronological order
                        # (latest first)
                        with feed_container.container():
                            st.markdown("### Live Moderated Feed")
                            for r in reversed(results):
                                color = "red" if r["label"] == "OFF" else "green"
                                st.markdown(
                                    f"**[{r['time']}]** &nbsp; "
                                    f":{color}[**{r['label']}** ({r['conf']:.2f})] &mdash; "
                                    f"\"{r['text']}\"")

                        # Small pause to allow Streamlit UI thread to render
                        await asyncio.sleep(0.05)

                status_box.success(
                    f"Stream finished. Moderated {limit} live posts successfully!")

        except Exception as e:
            status_box.error(f"WebSocket Connection Error: {e}")

    if st.button("Start Live Stream"):
        asyncio.run(run_bluesky_stream(stream_model, max_posts))


# ── Tab 8: Agentic Analysis ─────────────────────────────────────────────
with tab8:
    st.subheader("Agentic Website Audit (CrewAI)")
    st.markdown(
        "Leverage a CrewAI multi-agent workflow (Scraper Agent + Analyst Agent) "
        "to perform a deep, recursive web audit. The agents will crawl the site "
        "recursively (respecting robots.txt), classify all text elements, "
        "and produce a structured Markdown compliance audit report.")

    agent_url = st.text_input(
        "Enter Starting URL for Audit",
        value="https://example.com",
        key="agent_url")
    agent_model = st.selectbox(
        "Select LLM for Agents",
        ["gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "Ollama (Llama 3.1)"],
        key="agent_model",
        help="Select which model the agents will use. Ollama runs locally and bypasses rate limits."
    )
    # Prompt for Google API key if not set in environment
    agent_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get(
            "GOOGLE_API_KEY",
            ""),
        help="Required for CrewAI orchestration. Get a key from https://aistudio.google.com/apikey",
        key="agent_google_api_key")

    if st.button("Run Agentic Audit"):
        if not agent_url.strip():
            st.warning("Please enter a starting URL.")
        elif not agent_key.strip() and agent_model != "Ollama (Llama 3.1)":
            st.error(
                "Google API Key is required to run the agentic workflow using Gemini.")
        else:
            with st.spinner("CrewAI Agents are executing the audit... (this can take a minute)"):
                try:
                    from agents_workflow import run_agentic_audit
                    # Set GOOGLE_API_KEY in environment as CrewAI depends on it
                    os.environ["GOOGLE_API_KEY"] = agent_key
                    report = run_agentic_audit(
                        agent_url.strip(),
                        google_api_key=agent_key,
                        model_name=agent_model
                    )
                    st.success("Audit complete!")
                    st.markdown("### Executive Audit Report")
                    st.markdown(report)
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower(
                    ) or "resourceexhausted" in err_msg.lower():
                        st.error(
                            "**Gemini API quota exceeded** (free tier: 20 requests/day per model). "
                            "The agentic audit uses ~10-20 API calls per run.\n\n"
                            "**Options:**\n"
                            "1. Wait until tomorrow for quota reset\n"
                            "2. Try a different model (each model has a separate quota)\n"
                            "3. Upgrade to Gemini paid tier at https://ai.google.dev")
                    else:
                        st.error(f"Error during agentic audit: {e}")
