# ============================================================
# modules/explanation_generator.py
# Rule-based explanation + optional Groq LLM explanation
# ============================================================

import json
import requests

from config.settings import GROQ_API_KEY, GROQ_MODEL


# ── Rule-based ───────────────────────────────────────────────

def _rule_explanation(sentiment: str, pos_words: list, neg_words: list) -> str:
    """
    Build a human-readable sentence explaining the prediction
    based purely on keyword matches.
    """
    parts = []

    if sentiment == "positive":
        if pos_words:
            words = ", ".join(f"'{w}'" for w in pos_words[:5])
            parts.append(f"The text contains positive financial indicators such as {words}.")
        else:
            parts.append("The text has an overall positive financial tone.")
        if neg_words:
            words = ", ".join(f"'{w}'" for w in neg_words[:3])
            parts.append(f"Although some negative signals were also found ({words}), the positive signals dominated.")

    elif sentiment == "negative":
        if neg_words:
            words = ", ".join(f"'{w}'" for w in neg_words[:5])
            parts.append(f"The text contains negative financial indicators such as {words}.")
        else:
            parts.append("The text has an overall negative financial tone.")
        if pos_words:
            words = ", ".join(f"'{w}'" for w in pos_words[:3])
            parts.append(f"Some positive signals were also present ({words}), but the negative signals dominated.")

    else:  # neutral
        if pos_words or neg_words:
            pw = ", ".join(f"'{w}'" for w in pos_words[:3])
            nw = ", ".join(f"'{w}'" for w in neg_words[:3])
            parts.append(
                f"The text shows balanced signals — positive terms ({pw or 'none'}) "
                f"and negative terms ({nw or 'none'}) roughly cancel each other out."
            )
        else:
            parts.append("The text contains no strong positive or negative financial keywords, resulting in a neutral sentiment.")

    return " ".join(parts)


# ── Groq LLM ─────────────────────────────────────────────────

def _groq_explanation(text: str, sentiment: str) -> str:
    """
    Ask Groq to explain why the news is classified as the given sentiment.
    Returns the explanation string, or an error message on failure.
    """
    if not GROQ_API_KEY:
        return "(Groq API key not configured – LLM explanation unavailable.)"

    prompt = (
        f"The following financial news snippet has been classified as **{sentiment}** sentiment.\n\n"
        f"News: \"{text[:600]}\"\n\n"
        "In 2-3 sentences, explain why this news would be considered "
        f"{sentiment} from a financial perspective. Be concise and specific."
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a financial news analyst. Explain sentiment predictions briefly and clearly."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":   200,
                "temperature":  0.3,
            },
            timeout=15,
        )
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        return f"(Groq error: {data.get('error', {}).get('message', 'Unknown error')})"
    except requests.exceptions.Timeout:
        return "(Groq API timed out – LLM explanation unavailable.)"
    except Exception as exc:
        return f"(Groq API error: {exc})"


# ── Public entry point ────────────────────────────────────────

def generate_explanation(
    text: str,
    sentiment: str,
    pos_words: list,
    neg_words: list,
    use_groq: bool = True,
) -> tuple:
    """
    Generate both explanations and a combined summary.

    Returns
    -------
    (rule_explanation, llm_explanation, combined_explanation)
    """
    rule_exp = _rule_explanation(sentiment, pos_words, neg_words)
    llm_exp  = _groq_explanation(text, sentiment) if use_groq else ""

    if llm_exp and not llm_exp.startswith("("):
        combined = f"{rule_exp}\n\n[AI Analysis] {llm_exp}"
    else:
        combined = rule_exp

    return rule_exp, llm_exp, combined
