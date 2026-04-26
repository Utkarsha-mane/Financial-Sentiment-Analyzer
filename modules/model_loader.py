# ============================================================
# modules/model_loader.py  –  Load persisted model artefacts
# ============================================================

import os
import json
import warnings
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import (
    SVM_PATH, TFIDF_PATH, KEYWORDS_PATH, CSV_PATH, TFIDF_MAX_FEATURES
)


def _rebuild_tfidf() -> TfidfVectorizer:
    """
    The saved tfidf.pkl contains only hyper-parameters (no vocabulary).
    Re-fit on the same training corpus so the vocabulary is consistent
    with what the SVM expects.
    """
    print("[model_loader] Re-fitting TF-IDF on training corpus …")
    df = pd.read_csv(CSV_PATH).dropna(subset=["clean_news"])

    # Load the parameter template from the original (unfitted) pickle
    unfitted = joblib.load(os.path.join(os.path.dirname(TFIDF_PATH), "tfidf.pkl"))
    params = unfitted.get_params()
    params["max_features"] = TFIDF_MAX_FEATURES   # must equal 4999

    tfidf = TfidfVectorizer(**params)
    tfidf.fit(df["clean_news"])

    # Cache the fitted version so future runs are instant
    joblib.dump(tfidf, TFIDF_PATH)
    print(f"[model_loader] TF-IDF fitted ({len(tfidf.vocabulary_)} features). Cached.")
    return tfidf


def load_tfidf() -> TfidfVectorizer:
    """Load the fitted TF-IDF vectoriser (rebuild if necessary)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.exists(TFIDF_PATH):
            tfidf = joblib.load(TFIDF_PATH)
            # Quick sanity check – is it actually fitted?
            try:
                tfidf.transform(["test"])
                return tfidf
            except Exception:
                pass
        return _rebuild_tfidf()


def load_svm():
    """Load the trained LinearSVC model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(SVM_PATH)


def load_keywords() -> dict:
    """Load positive / negative keyword lists."""
    with open(KEYWORDS_PATH, "r") as f:
        return json.load(f)
