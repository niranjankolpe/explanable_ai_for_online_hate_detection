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
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Tags whose content is not visible text — strip them entirely
_STRIP_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "meta",
    "link"}

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


class RecursiveCrawler:
    """
    Recursive web crawler that follows internal links up to a depth limit,
    obeys robots.txt directives, and restricts crawling to the original start domain.
    """

    def __init__(
            self,
            max_depth: int = 2,
            max_pages: int = 10,
            user_agent: str = "*"):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.user_agent = user_agent
        self.visited = set()
        self.robots_parsers = {}  # Cache parsed robots.txt by domain

    def _get_robots_parser(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if base_url not in self.robots_parsers:
            rp = RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            try:
                # Add headers matching our standard requests
                # Note: RobotFileParser reads natively via urllib, so we just
                # set read timeout
                rp.read()
            except Exception:
                rp = None
            self.robots_parsers[base_url] = rp
        return self.robots_parsers[base_url]

    def is_allowed(self, url: str) -> bool:
        rp = self._get_robots_parser(url)
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def extract_links(self, url: str, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(url, href)
            # Remove fragment/anchor tags
            full_url = full_url.split("#")[0]

            parsed_full = urlparse(full_url)
            # Only traverse HTTP/HTTPS links on the same start domain
            if parsed_full.netloc == base_domain and parsed_full.scheme in {
                    "http", "https"}:
                links.append(full_url)
        return list(set(links))

    def crawl(self, start_url: str) -> dict:
        results = {}
        queue = [(start_url, 0)]  # (url, depth)

        while queue and len(self.visited) < self.max_pages:
            url, depth = queue.pop(0)

            if url in self.visited:
                continue

            # Enforce robots.txt rules
            if not self.is_allowed(url):
                results[url] = {
                    "status": "error",
                    "error": "Disallowed by robots.txt"}
                continue

            self.visited.add(url)

            try:
                # Single HTTP request per URL — reuse response for both
                # text extraction and link discovery
                resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    results[url] = {
                        "status": "error",
                        "error": f"Non-text content type: {content_type}"}
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove non-visible elements (same logic as scrape_text)
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

                results[url] = {"status": "ok", "texts": chunks}

                # Queue child links if below max depth (reuses same resp.text)
                if depth < self.max_depth:
                    child_links = self.extract_links(url, resp.text)
                    for link in child_links:
                        if link not in self.visited:
                            queue.append((link, depth + 1))

            except Exception as e:
                results[url] = {"status": "error", "error": str(e)}

        return results
