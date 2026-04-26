# ============================================================
# config/settings.py  –  Application-wide configuration
# ============================================================

import os
from pathlib import Path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\UTKARSHA\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# load local .env file if present
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if load_dotenv is not None:
    load_dotenv(Path(BASE_DIR) / ".env")

# ── Groq API ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # set via env or paste here
GROQ_MODEL   = "llama3-8b-8192"                      # fast, free-tier model

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_PATH     = os.path.join(BASE_DIR, "sentiment_results.db")

SVM_PATH        = os.path.join(MODEL_DIR, "svm_model.pkl")
TFIDF_PATH      = os.path.join(MODEL_DIR, "tfidf_fitted.pkl")  # refitted copy
KEYWORDS_PATH   = os.path.join(MODEL_DIR, "keywords.json")
CSV_PATH        = os.path.join(DATA_DIR,  "processed_data.csv")

# ── TF-IDF rebuild params (must match training) ───────────────
TFIDF_MAX_FEATURES = 4999   # 4999 tfidf + 4 keyword cols = 5003 (SVM input)

# ── Financial-domain check words ────────────────────────────
FINANCIAL_KEYWORDS = [
    "stock", "share", "market", "profit", "loss", "revenue", "earnings",
    "dividend", "ipo", "acquisition", "merger", "fiscal", "quarter",
    "investor", "trading", "exchange", "bond", "equity", "hedge",
    "portfolio", "gdp", "inflation", "recession", "bank", "financial",
    "economy", "economic", "fund", "asset", "debt", "capital", "nasdaq",
    "nyse", "dow", "s&p", "currency", "forex", "commodity", "oil",
    "gold", "interest rate", "fed", "central bank", "analyst", "forecast",
    "guidance", "outlook", "growth", "decline", "rise", "fall", "surge",
    "plunge", "rally", "sell", "buy", "investment", "return", "yield",
]
