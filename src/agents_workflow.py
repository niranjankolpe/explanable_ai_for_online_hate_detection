"""
agents_workflow.py
CrewAI multi-agent workflow for automated Hate Speech Content Auditing.

Defines Scraper and Analyst agents, custom tools, and orchestrator function.
"""

import os
import tempfile
import pandas as pd
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import PrivateAttr
from crewai import Agent, Task, Crew
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from typing import Any

from crawler import RecursiveCrawler
from predict import load_model, predict_proba, get_label_conf

load_dotenv()


class LangChainLLM(BaseLLM):
    """
    Custom wrapper that exposes a LangChain LLM as a CrewAI BaseLLM.
    This bypasses LiteLLM and native SDK connection/validation errors.
    """
    model: str = "gemini-3.5-flash"
    _lc_llm: Any = PrivateAttr(default=None)

    def __init__(self, lc_llm: Any, **kwargs):
        model_name = getattr(lc_llm, "model", "llama3.1:8b")
        super().__init__(model=model_name, **kwargs)
        object.__setattr__(self, "_lc_llm", lc_llm)

    def call(self, messages, tools=None, callbacks=None, **kwargs):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        lc_msgs = []
        if isinstance(messages, str):
            lc_msgs.append(HumanMessage(content=messages))
        else:
            for m in messages:
                role = getattr(
                    m, "role", None) or m.get("role") if hasattr(
                    m, "get") else None
                content = getattr(m, "content", None) or m.get(
                    "content") if hasattr(m, "get") else str(m)
                if role == "system":
                    lc_msgs.append(SystemMessage(content=content))
                elif role in ("assistant", "model"):
                    lc_msgs.append(AIMessage(content=content))
                else:
                    lc_msgs.append(HumanMessage(content=content))
        res = self._lc_llm.invoke(lc_msgs)
        content = res.content
        if isinstance(content, list):
            # Sometimes Gemini returns a list of dictionaries for text parts
            if len(content) > 0 and isinstance(
                    content[0], dict) and 'text' in content[0]:
                content = "\n".join([c.get('text', '')
                                    for c in content if isinstance(c, dict)])
            else:
                content = str(content)
        return str(content)


# Global variable or cache for loaded models to avoid reloading
_MODEL_CACHE = {}


def _get_cached_model(model_type: str, subtask: str = "a"):
    key = (model_type, subtask)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(model_type, subtask)
    return _MODEL_CACHE[key]


@tool("Crawl Website Recursively")
def crawl_website(url: str) -> str:
    """
    Crawls a website recursively starting from url up to max_pages=10 and max_depth=2.
    Respects robots.txt and remains on the starting domain.
    Returns a formatted text string summarizing all scraped pages and their raw content.
    """
    try:
        crawler = RecursiveCrawler(max_depth=2, max_pages=10)
        results = crawler.crawl(url)

        output_lines = []
        for page_url, data in results.items():
            output_lines.append(f"=== PAGE: {page_url} ===")
            if data["status"] == "ok":
                texts = data.get("texts", [])
                if texts:
                    output_lines.append("\n".join(texts))
                else:
                    output_lines.append("(No significant text extracted)")
            else:
                output_lines.append(
                    f"Error crawling page: {data.get('error', 'unknown error')}")
            output_lines.append("\n")

        return "\n".join(output_lines)
    except Exception as e:
        return f"Error during crawl execution: {str(e)}"


@tool("Classify Extracted Content")
def classify_content(text_blocks_raw: str) -> str:
    """
    Parses pages and classifies text content using the local Hate Speech classifier.
    Expects a raw text block containing PAGE headings and content.
    Returns a summary of predictions, offense rates, and identified high-risk content.
    """
    try:
        # Simple parser to split by PAGE headings
        lines = text_blocks_raw.split("\n")
        current_url = "unknown"
        page_texts = {}

        for line in lines:
            if line.startswith("=== PAGE: ") and line.endswith(" ==="):
                current_url = line[10:-4]
                page_texts[current_url] = []
            elif current_url != "unknown" and line.strip():
                page_texts[current_url].append(line.strip())

        # Load the baseline classifier model (subtask a)
        model, aux = _get_cached_model("baseline", "a")

        output_reports = []

        total_offensive = 0
        total_checked = 0

        for url, texts in page_texts.items():
            if not texts:
                continue
            # Predict probabilities
            probas = predict_proba(texts, "baseline", model, aux, subtask="a")

            page_off_count = 0
            page_results = []

            for text, proba in zip(texts, probas):
                label, conf = get_label_conf(proba, "a")
                total_checked += 1
                if label == "OFF":
                    page_off_count += 1
                    total_offensive += 1
                    page_results.append(
                        f"- [OFFENSIVE] (Conf: {conf:.1%}): \"{text}\"")
                else:
                    page_results.append(
                        f"- [NOT OFFENSIVE] (Conf: {conf:.1%}): \"{text}\"")

            off_pct = (page_off_count / len(texts)) * 100 if texts else 0
            output_reports.append(f"Page: {url}")
            output_reports.append(
                f"Offensive Content Rate: {off_pct:.1f}% ({page_off_count}/{len(texts)} chunks)")
            # limit output size
            output_reports.append("\n".join(page_results[:10]))
            if len(page_results) > 10:
                output_reports.append(
                    f"... and {len(page_results) - 10} more chunks")
            output_reports.append("-" * 40)

        summary = (
            f"Audit Summary:\n"
            f"Total Checked: {total_checked}\n"
            f"Total Offensive: {total_offensive}\n"
            f"Overall Offense Rate: {(total_offensive / total_checked * 100) if total_checked > 0 else 0:.1f}%\n\n"
        )
        return summary + "\n".join(output_reports)
    except Exception as e:
        return f"Error during classification: {str(e)}"


def run_agentic_audit(
        start_url: str,
        google_api_key: str = None,
        model_name: str = "gemini-2.0-flash") -> str:
    """
    Orchestrates the CrewAI agents to scrape and audit the website content.
    Returns the final markdown report from the Analyst Agent.
    """
    if model_name == "Ollama (Llama 3.1)":
        if ChatOllama is None:
            raise ImportError(
                "langchain-ollama is not installed. Please install it to use Ollama.")
        lc_llm = ChatOllama(model="llama3.1:8b", temperature=0.2)
    else:
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Google API Key is required for CrewAI workflow execution using Gemini.")

        lc_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )

    llm = LangChainLLM(lc_llm)

    # 1. Scraper Agent
    scraper = Agent(
        role="Website Content Scraper",
        goal=f"Scrape the website starting at {start_url} and extract all readable text content.",
        backstory="An automated expert web crawler and scraper, specializing in retrieving clean, structured text data from website hierarchies while strictly respecting robots.txt.",
        tools=[crawl_website],
        llm=llm,
        verbose=True,
    )

    # 2. Analyst Agent
    analyst = Agent(
        role="Hate Speech Content Auditor",
        goal="Process crawled web page texts, classify chunks for offensive speech, and write an audit report.",
        backstory="A specialized Trust & Safety analyst. You consume crawled website content, classify it using local models, identify potential hate speech violations, and write executive-level compliance reports.",
        tools=[classify_content],
        llm=llm,
        verbose=True,
    )

    # Task 1: Scraping
    scrape_task = Task(
        description=f"Crawl and harvest text content recursively starting from: {start_url}. Deliver the raw crawler text content grouped by URL.",
        expected_output="A structured raw text log of crawled pages and their corresponding text blocks.",
        agent=scraper,
    )

    # Task 2: Analyzing
    audit_task = Task(
        description=(
            "Review the raw crawled texts, run the content classification tool, and generate a final compliance and hate speech audit report. "
            "Your report must contain:\n"
            "1. An executive summary with statistics (total pages crawled, total text blocks, offense rate, overall risk level).\n"
            "2. A breakdown per URL with their respective offense rates.\n"
            "3. Examples of flagged offensive sentences, highlighting high-risk segments.\n"
            "4. Actionable recommendations for trust & safety moderation or filtering."),
        expected_output="A comprehensive Markdown compliance audit report listing statistics, breakdowns, flagged examples, and mitigation advice.",
        agent=analyst,
    )

    crew = Crew(
        agents=[scraper, analyst],
        tasks=[scrape_task, audit_task],
        verbose=True,
        max_rpm=15,
        max_iter=2,
        memory=False,
    )

    result = crew.kickoff()

    # Handle CrewAI output wrapper variations
    if hasattr(result, "raw"):
        return result.raw
    return str(result)
