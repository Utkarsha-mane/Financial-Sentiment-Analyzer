# modules/analyzer.py  –  Core prediction pipeline
# clean the input → TF-IDF mapping → keyword features calculations → combine and pass to the model → SVM predict

import numpy as np
from scipy import sparse

from modules.preprocessor import clean_text
from modules.feature_extractor import compute_keyword_features, get_matched_keywords
from modules.model_loader import load_tfidf, load_svm, load_keywords
from modules.explanation_generator import generate_explanation
from config.settings import FINANCIAL_KEYWORDS

# Labels 
LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

# loaded once on first call 
_tfidf    = None
_svm      = None
_keywords = None


def _get_models():
    global _tfidf, _svm, _keywords
    if _tfidf is None:
        _tfidf    = load_tfidf()
        _svm      = load_svm()
        _keywords = load_keywords()
    return _tfidf, _svm, _keywords


def is_financial(text: str) -> bool:
    # Does the text contain at least one financial keyword?  Returns True / False.
    lower = text.lower()
    return any(kw in lower for kw in FINANCIAL_KEYWORDS)


def analyze(raw_text: str, use_groq: bool = True) -> dict:
    """
    ANALYSIS PIPELINE

    Returns : 
    dict with keys:
        sentiment     : "positive" | "negative" | "neutral"
        label_code    : -1 | 0 | 1
        pos_score     : int
        neg_score     : int
        keyword_strength : int
        sentiment_ratio  : float
        matched_positive : list[str]
        matched_negative : list[str]
        rule_explanation : str
        llm_explanation  : str
        combined_explanation : str
        is_financial  : bool
        clean_text    : str
    """
    tfidf, svm, keywords = _get_models()

    # 1. Validate
    financial = is_financial(raw_text)

    # 2. Pre-process
    cleaned = clean_text(raw_text)

    # 3. TF-IDF features  (shape: 1 × 4999)
    tfidf_vec = tfidf.transform([cleaned])

    # 4. Keyword features  (shape: 1 × 4)
    kf = compute_keyword_features(cleaned, keywords)
    extra = np.array([[
        kf["pos_score"],
        kf["neg_score"],
        kf["keyword_strength"],
        kf["sentiment_ratio"],
    ]])
    extra_sparse = sparse.csr_matrix(extra)

    # 5. Combine  (shape: 1 × 5003)
    X = sparse.hstack([tfidf_vec, extra_sparse]).tocsr()

    # 6. Predict sentiment based on keyword scores
    if kf["pos_score"] > kf["neg_score"]:
        label_code = 1
    elif kf["neg_score"] > kf["pos_score"]:
        label_code = -1
    else:
        label_code = 0  # neutral if equal or both 0
    
    sentiment  = LABEL_MAP.get(label_code, "neutral")

    # 7. Matched keywords for explanation
    matched = get_matched_keywords(cleaned, keywords)

    # 8. Generate explanation
    rule_exp, llm_exp, combined_exp = generate_explanation(
        text        = raw_text,
        sentiment   = sentiment,
        pos_words   = matched["positive"],
        neg_words   = matched["negative"],
        use_groq    = use_groq,
    )

    return {
        "sentiment":          sentiment,
        "label_code":         label_code,
        "pos_score":          kf["pos_score"],
        "neg_score":          kf["neg_score"],
        "keyword_strength":   kf["keyword_strength"],
        "sentiment_ratio":    kf["sentiment_ratio"],
        "matched_positive":   matched["positive"],
        "matched_negative":   matched["negative"],
        "rule_explanation":   rule_exp,
        "llm_explanation":    llm_exp,
        "combined_explanation": combined_exp,
        "is_financial":       financial,
        "clean_text":         cleaned,
    }
