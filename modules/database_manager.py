# modules/database_manager.py  –  SQLite persistence layer

import sqlite3
import datetime
from config.settings import DB_PATH


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # Create the results table if it does not already exist.
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            source_type      TEXT    NOT NULL,   -- 'text' | 'image' | 'url'
            raw_text         TEXT,
            clean_text       TEXT,
            sentiment        TEXT    NOT NULL,
            label_code       INTEGER NOT NULL,
            pos_score        INTEGER,
            neg_score        INTEGER,
            keyword_strength INTEGER,
            sentiment_ratio  REAL,
            explanation      TEXT,
            is_financial     INTEGER             -- 0 or 1
        )
    """)
    conn.commit()
    conn.close()


def save_result(result: dict, source_type: str = "text", raw_text: str = ""):
    # Persist a single analysis result to the database.
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO results
            (timestamp, source_type, raw_text, clean_text, sentiment, label_code,
             pos_score, neg_score, keyword_strength, sentiment_ratio,
             explanation, is_financial)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_type,
            raw_text[:2000],
            result.get("clean_text", "")[:2000],
            result.get("sentiment", ""),
            result.get("label_code", 0),
            result.get("pos_score", 0),
            result.get("neg_score", 0),
            result.get("keyword_strength", 0),
            result.get("sentiment_ratio", 0.0),
            result.get("combined_explanation", "")[:1000],
            int(result.get("is_financial", False)),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all() -> list:
    # Return all stored results as a list of dicts.
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM results ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_recent(n: int = 50) -> list:
    # Return the n most recent results.
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM results ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_db():
    # Delete all records.
    conn = _get_connection()
    conn.execute("DELETE FROM results")
    conn.commit()
    conn.close()
