# modules/url_processor.py  –  Extract article text from a URL
# Primary: newspaper3k   |   Fallback: requests + BeautifulSoup

import re
import requests
from config.settings import FINANCIAL_KEYWORDS


# Primary extractor 

def _extract_via_newspaper(url: str) -> str:
    # Use newspaper3k Article for clean article text. Returns empty string on failure.
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""


# Fallback extractor 

def _extract_via_bs4(url: str) -> str:
    # Fallback: fetch raw HTML with requests and strip tags using BeautifulSoup.
    try:
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script / style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Prefer article or main body content
        body = soup.find("article") or soup.find("main") or soup.body
        if body:
            text = body.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


# Financial domain check 

def _is_financial_url_content(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in FINANCIAL_KEYWORDS)


# Main function 
def extract_text_from_url(url: str) -> tuple:
    # Download and extract the main article text from a URL.
    # Returns a tuple of (extracted_text, is_financial_boolean).

    # Try newspaper3k first
    text = _extract_via_newspaper(url)

    # Fall back to BeautifulSoup if newspaper returns nothing
    if not text or len(text) < 100:
        text = _extract_via_bs4(url)

    if not text:
        return "", False

    financial = _is_financial_url_content(text)
    return text, financial
