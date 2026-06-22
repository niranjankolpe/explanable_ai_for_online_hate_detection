"""
crawler.py
Web scraping module for collecting text data from URLs.

Scrapes visible text content from web pages using BeautifulSoup,
supports bulk URL processing, and saves results to CSV.

Usage:
    from crawler import scrape_text, scrape_multiple, save_crawled_data

    texts = scrape_text("https://example.com")
    results = scrape_multiple(["https://a.com", "https://b.com"])
    save_crawled_data(results, "crawled_data/output.csv")
"""

import os
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Tags whose content is not visible text — strip them entirely
_STRIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas", "meta", "link"}

# Tags that typically hold readable content
_CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th",
                 "blockquote", "article", "section", "figcaption", "dd", "dt"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

_TIMEOUT = 10  # seconds


def scrape_text(url: str) -> list[str]:
    """
    Fetch a URL and extract visible text content.

    Returns a list of text chunks (paragraphs, headings, list items, etc.).
    Filters out chunks shorter than 3 words.
    Raises on HTTP errors or connection failures.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError(f"Non-text content type: {content_type}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-visible elements
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()

    chunks = []
    for tag in soup.find_all(_CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 3:
            chunks.append(text)

    return chunks


def scrape_multiple(urls: list[str]) -> dict:
    """
    Scrape text from a list of URLs.

    Returns a dict mapping each URL to either:
      - a list of text chunks (on success)
      - an error string (on failure)
    """
    results = {}
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            chunks = scrape_text(url)
            results[url] = {"status": "ok", "texts": chunks}
        except Exception as e:
            results[url] = {"status": "error", "error": str(e)}
    return results


def save_crawled_data(results: dict, output_dir: str = "crawled_data") -> str:
    """
    Save crawled text to a timestamped CSV file.

    Returns the path to the saved CSV file.
    Columns: url, text, scraped_at
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    now = datetime.now().isoformat()
    for url, data in results.items():
        if data["status"] != "ok":
            continue
        for text in data["texts"]:
            rows.append({"url": url, "text": text, "scraped_at": now})

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{timestamp}_crawled.csv")
    df.to_csv(path, index=False)
    return path
