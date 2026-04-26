# ============================================================
# modules/preprocessor.py  –  Text cleaning pipeline
# Must replicate the same steps used during training.
# ============================================================

import re
import string


def clean_text(text: str) -> str:
    """
    Clean raw text to match the preprocessing applied during training.

    Steps:
        1. Lower-case
        2. Remove URLs
        3. Remove punctuation
        4. Collapse multiple whitespace characters
        5. Strip leading / trailing whitespace

    Parameters
    ----------
    text : str  – raw news text

    Returns
    -------
    str  – cleaned text ready for TF-IDF vectorisation
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lower-case
    text = text.lower()

    # 2. Remove URLs (http/https/www)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Remove punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Remove digits (uncomment if training did this)
    # text = re.sub(r"\d+", " ", text)

    # 5. Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()
