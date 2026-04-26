# ============================================================
# modules/feature_extractor.py  –  Keyword-based feature computation
# Produces the 4 extra columns appended after TF-IDF features.
# ============================================================

import json
from typing import Dict


def load_keywords(path: str) -> Dict:
    """Load positive/negative keyword lists from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_keyword_features(text: str, keywords: Dict) -> Dict:
    """
    Compute four keyword-derived features for a piece of text.

    Features
    --------
    pos_score        : count of positive keywords found in text
    neg_score        : count of negative keywords found in text
    keyword_strength : pos_score + neg_score  (total signal strength)
    sentiment_ratio  : pos_score / (neg_score + 1)  (ratio, +1 avoids /0)

    Parameters
    ----------
    text     : str  – cleaned (lower-cased) text
    keywords : dict – {"positive": [...], "negative": [...]}

    Returns
    -------
    dict with the four feature values
    """
    tokens = set(text.lower().split())

    positive_words = keywords.get("positive", [])
    negative_words = keywords.get("negative", [])

    # Count keyword matches
    pos_score = sum(1 for w in positive_words if w in tokens)
    neg_score = sum(1 for w in negative_words if w in tokens)

    keyword_strength = pos_score + neg_score
    sentiment_ratio  = pos_score / (neg_score + 1)

    return {
        "pos_score":        pos_score,
        "neg_score":        neg_score,
        "keyword_strength": keyword_strength,
        "sentiment_ratio":  round(sentiment_ratio, 4),
    }


def get_matched_keywords(text: str, keywords: Dict) -> Dict:
    """
    Return the actual matched keyword words (used for explanation).

    Returns
    -------
    dict {"positive": [word, ...], "negative": [word, ...]}
    """
    tokens = set(text.lower().split())
    matched_pos = [w for w in keywords.get("positive", []) if w in tokens]
    matched_neg = [w for w in keywords.get("negative", []) if w in tokens]
    return {"positive": matched_pos, "negative": matched_neg}
