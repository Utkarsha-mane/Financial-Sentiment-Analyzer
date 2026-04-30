# config/settings.py  –  all configurations are in this file

import os
from pathlib import Path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# load local .env file 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if load_dotenv is not None:
    load_dotenv(str(Path(BASE_DIR) / ".env"))

# Groq API 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   
GROQ_MODEL   = "llama-3.3-70b-versatile"  # Currently available model

# Paths 
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_PATH     = os.path.join(BASE_DIR, "sentiment_results.db")

SVM_PATH        = os.path.join(MODEL_DIR, "svm_model.pkl")
TFIDF_PATH      = os.path.join(MODEL_DIR, "tfidf_fitted.pkl")  
KEYWORDS_PATH   = os.path.join(MODEL_DIR, "keywords.json")
CSV_PATH        = os.path.join(DATA_DIR,  "processed_data.csv")

# TF-IDF rebuild parameters  
TFIDF_MAX_FEATURES = 4999   # 4999 tfidf + 4 keyword cols = 5003 (SVM input)

# Financial-domain check words 
FINANCIAL_KEYWORDS = [
    "stock", "share", "market", "profit", "loss", "revenue", "earnings",
    "dividend", "ipo", "acquisition", "merger", "fiscal", "quarter",
    "investor", "trading", "exchange", "bond", "equity", "hedge",
    "portfolio", "gdp", "inflation", "recession", "bank", "financial",
    "economy", "economic", "fund", "asset", "debt", "capital", 
    "dow", "s&p", "currency", "forex", "commodity", "oil",
    "gold", "interest rate", "fed", "central bank", "analyst", "forecast",
    "guidance", "outlook", "growth", "decline", "rise", "fall", "surge",
    "plunge", "rally", "sell", "buy", "investment", "return", "yield",
    "ebit", "ebitda", "operating income", "net income", "gross margin",
    "operating margin", "cash flow", "free cash flow", "balance sheet",
    "income statement", "valuation", "book value", "write-off",
    "impairment", "amortization", "depreciation", "liquidity",
    "bull market", "bear market", "volatility", "index", "benchmark",
    "nifty", "sensex", "nasdaq", "dow jones", "ftse",
    "trading volume", "market cap", "price action", "correction",
    "breakout", "support", "resistance", "short selling", "long position",
    "intraday", "derivatives", "futures", "options", "strike price",
    "mutual fund", "etf", "hedge fund", "private equity", "venture capital",
    "angel investor", "seed funding", "series a", "series b",
    "funding round", "capital raise", "allocation", "diversification",
    "loan", "credit", "interest", "interest rate hike", "rate cut",
    "repo rate", "reverse repo", "npa", "default", "credit rating",
    "collateral", "mortgage", "lending", "borrowing",
    "gdp growth", "cpi", "wpi", "deflation", "stagflation",
    "monetary policy", "fiscal policy", "stimulus", "subsidy",
    "budget", "deficit", "surplus", "trade deficit",
    "import", "export", "tariff", "sanctions",
    "risk", "volatility index", "drawdown", "returns", "roi",
    "alpha", "beta", "sharpe ratio", "performance", "underperform",
    "outperform", "guidance cut", "downgrade", "upgrade",
    "buyback", "stock split", "bonus shares", "rights issue",
    "delisting", "listing", "spin-off", "takeover", "restructuring",
    "crude oil", "brent", "natural gas", "metal", "silver",
    "exchange rate", "currency depreciation", "appreciation",
    "usd", "inr", "eur", "yen",
    "beat estimates", "missed estimates", "strong results",
    "weak outlook", "record high", "all-time low",
    "market sentiment", "investor confidence",
    "economic slowdown", "recovery", "expansion", "contraction",
    "guidance raised", "guidance lowered"
]
