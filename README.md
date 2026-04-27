# Financial News Sentiment Analyzer

A desktop application for three-class sentiment analysis of financial news text. Built with Python and Tkinter, the system combines a trained Linear SVM classifier with keyword-based scoring and optional LLM-generated explanations via the Groq API. Input is accepted as raw text, image (OCR), or a URL. Results are persisted to a local SQLite database and visualized through a Matplotlib dashboard.

---

## Architecture Overview

```
project/
├── main.py                        # Entry point
├── requirements.txt
├── config/
│   └── settings.py                # Paths, API keys, TFIDF params, financial keyword list
├── model/
│   ├── svm_model.pkl              # Trained LinearSVC (3-class: -1, 0, 1)
│   ├── tfidf.pkl                  # TF-IDF parameter template (unfitted)
│   ├── tfidf_fitted.pkl           # TF-IDF fitted on training corpus (auto-generated)
│   └── keywords.json              # Domain keyword lists (positive/negative)
├── data/
│   └── processed_data.csv         # Preprocessed training corpus (4,839 records)
├── modules/
│   ├── preprocessor.py            # Text cleaning (lowercase, URL strip, punctuation removal)
│   ├── feature_extractor.py       # Keyword feature computation
│   ├── model_loader.py            # Model loading and TF-IDF rebuild logic
│   ├── analyzer.py                # Full prediction pipeline
│   ├── image_processor.py         # OpenCV preprocessing + pytesseract OCR
│   ├── url_processor.py           # Article extraction via newspaper3k / BeautifulSoup
│   ├── visualizer.py              # Matplotlib 4-chart dashboard
│   ├── database_manager.py        # SQLite schema, read/write operations
│   └── explanation_generator.py   # Rule-based + Groq LLM explanation generation
└── ui/
    └── app_gui.py                 # Tkinter GUI (tabbed input, results panel, action bar)
```

---

## Machine Learning Pipeline

### Dataset
- Source: `data/processed_data.csv`
- Records: 4,839 financial news headlines and articles
- Class distribution: Neutral 2,872 / Positive 1,363 / Negative 604
- Columns: `sentiment`, `news`, `clean_news`, `target`, `pos_score`, `neg_score`, `keyword_strength`, `sentiment_ratio`

### Feature Engineering
The model uses a combined feature matrix of 5,003 dimensions:

**TF-IDF features (4,999 dimensions)**
- Vectorizer: `TfidfVectorizer` with `max_features=4999`, `ngram_range=(1,2)`, `max_df=0.85`, `min_df=2`, `sublinear_tf=True`
- Note: `tfidf.pkl` was serialized without a fitted vocabulary. On first run, `model_loader.py` re-fits the vectorizer on `processed_data.csv` and caches the result as `tfidf_fitted.pkl`.

**Keyword features (4 dimensions)**
These four scalar features are appended to the TF-IDF vector before prediction:

| Feature | Computation |
|---|---|
| `pos_score` | Count of positive keyword matches in tokenized text |
| `neg_score` | Count of negative keyword matches in tokenized text |
| `keyword_strength` | `pos_score + neg_score` |
| `sentiment_ratio` | `pos_score / (neg_score + 1)` |

### Classifier
- Model: `LinearSVC` (scikit-learn)
- Classes: `-1` (negative), `0` (neutral), `1` (positive)
- Input dimensionality: 5,003 features
- Stored at: `model/svm_model.pkl`

### Keyword Override Logic
Because `tfidf.pkl` was saved without its fitted vocabulary and must be re-fitted, the SVM predictions can be unreliable where keyword scores strongly contradict the prediction. `analyzer.py` applies a post-prediction correction:

- If `neg_score > pos_score`, prediction is overridden to **negative**
- If `pos_score > neg_score`, prediction is overridden to **positive**
- Else, prediction is overridden to **neutral**

---

## Module Reference

### preprocessor.py
`clean_text(text)` applies lowercase conversion, URL removal (`http`, `https`, `www`), punctuation stripping via `str.translate`, and whitespace normalization. Must remain consistent with the preprocessing applied during original training.

### feature_extractor.py
`compute_keyword_features(text, keywords)` returns the four keyword scalar features as a dict. `get_matched_keywords(text, keywords)` returns the actual matched words split by polarity, used by the explanation generator.

### model_loader.py
`load_tfidf()` loads `tfidf_fitted.pkl` if it exists and passes a transform sanity check. If not, re-fits on `processed_data.csv` using parameters extracted from `tfidf.pkl` and writes the result to `tfidf_fitted.pkl`. `load_svm()` loads `svm_model.pkl`. `load_keywords()` loads `keywords.json`. All three are lazy-loaded singletons in `analyzer.py` and initialized once per session.

### analyzer.py
`analyze(raw_text, use_groq)` runs the full pipeline: clean text, transform via TF-IDF, compute keyword features, combine into a sparse matrix of shape `(1, 5003)`, predict with SVM, apply keyword override, generate explanations. Returns a dict containing sentiment label, label code, all four keyword scores, matched keyword lists, rule explanation, LLM explanation, combined explanation, financial domain flag, and cleaned text.

`is_financial(text)` checks whether the text contains at least one token from the financial domain word list defined in `config/settings.py`.

### image_processor.py
`extract_text_from_image(path)` reads the image with OpenCV, converts to grayscale, applies `fastNlMeansDenoising` followed by `adaptiveThreshold` (Gaussian, block size 11), then runs pytesseract with `--oem 3 --psm 3`. Falls back to PIL for formats that OpenCV cannot decode.

### url_processor.py
`extract_text_from_url(url)` attempts extraction via `newspaper3k` first. Falls back to `requests` + `BeautifulSoup` if the result is empty or under 100 characters. The fallback strips `script`, `style`, `nav`, `footer`, and `header` tags and prefers `article` or `main` elements. Returns the article body string and a boolean financial domain flag.

### explanation_generator.py
`generate_explanation(text, sentiment, pos_words, neg_words, use_groq)` produces three strings: a rule-based explanation constructed from matched keyword lists, a Groq API explanation using `llama3-8b-8192` with `max_tokens=200` and `temperature=0.3`, and a combined output that appends the LLM result when available. If the API key is absent or the call fails, only the rule-based string is returned with no exception raised.

### database_manager.py
SQLite database written to `sentiment_results.db` in the project root. The `results` table stores: `timestamp`, `source_type` (`text` / `image` / `url`), `raw_text`, `clean_text`, `sentiment`, `label_code`, `pos_score`, `neg_score`, `keyword_strength`, `sentiment_ratio`, `explanation`, `is_financial`. Table is created automatically on first run via `init_db()`. No manual setup required.

### visualizer.py
`show_dashboard()` fetches all records from the database and renders a 2x2 Matplotlib figure containing: a bar chart of sentiment class counts, a pie chart of sentiment proportions, a horizontal diverging bar chart of top positive and negative keyword frequencies across all saved records, and a line chart of sentiment over insertion order with a rolling average overlay. Requires at least one saved record.

### app_gui.py
Tkinter application window (1050x780, dark theme). Left panel: tabbed notebook with Text, URL, and Image input tabs. Right panel: sentiment badge, score table (four keyword metrics), matched keywords display, and scrollable explanation field. Bottom action bar: Analyze, Show Charts, and Save to Database buttons. Analysis executes in a background daemon thread via `threading.Thread`. A boolean `_analyzing` flag prevents concurrent analysis calls. Errors during analysis are printed to the terminal and shown in a dialog.

---

## Setup

### 1. Create and activate a virtual environment

**Windows**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

Required only for the Image input tab. The rest of the application works without it.

**Windows** — Download the installer from https://github.com/UB-Mannheim/tesseract/wiki. After installation, either add the install directory to the system PATH or set the binary path explicitly in `config/settings.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**macOS**
```bash
brew install tesseract
```

**Ubuntu / Debian**
```bash
sudo apt-get install tesseract-ocr
```

### 4. Configure the Groq API key

LLM explanations require a free Groq API key, available at https://console.groq.com. The application functions fully without it — rule-based explanations are always generated regardless.

Set via environment variable:
```bash
export GROQ_API_KEY="gsk_..."        # macOS / Linux
set GROQ_API_KEY=gsk_...             # Windows CMD
```

Or set directly in `config/settings.py`:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### 5. Run

```bash
python main.py
```

On the first run, `tfidf_fitted.pkl` is generated from `processed_data.csv` and cached in `model/`. All subsequent runs load the cached file directly.

---

## Usage

**Text input** — Paste financial news text into the Text tab and click Analyze Text. Clear the field with the Clear button before entering a new sample.

**URL input** — Enter a full news article URL including the `https://` scheme. The app fetches and extracts the article body, displays a preview, and runs analysis on the extracted text. Requires internet access.

**Image input** — Click Browse Image to select a PNG, JPG, BMP, or TIFF file. Click Analyze Image to run OCR and then analysis. Requires Tesseract to be installed and accessible.

**Saving results** — After each analysis, click Save to Database. Results are not saved automatically. The Save button activates only after a successful analysis completes.

**Dashboard** — Click Show Charts / Dashboard after saving at least one result. The trend chart becomes meaningful with five or more saved records of mixed sentiment.

---

## Dependencies

| Package | Purpose |
|---|---|
| scikit-learn | LinearSVC classifier, TfidfVectorizer |
| pandas | Dataset loading |
| numpy | Array operations |
| scipy | Sparse matrix construction |
| joblib | Model deserialization |
| opencv-python | Image preprocessing |
| pytesseract | OCR interface |
| Pillow | Image format fallback for OpenCV |
| newspaper3k | Primary article extraction |
| beautifulsoup4 | HTML parsing fallback |
| lxml | HTML parser backend for BeautifulSoup |
| requests | HTTP client for URL fetch and Groq API |
| matplotlib | Dashboard charts |
| tkinter | GUI framework (included with Python) |

---

## Known Limitations

`tfidf.pkl` was serialized without a fitted vocabulary. The vectorizer is re-fitted on `processed_data.csv` at runtime, which does not guarantee an identical vocabulary to the one used during original model training. This mismatch reduces SVM classification accuracy on some inputs. The keyword override in `analyzer.py` compensates for cases with strong unambiguous keyword signals. On inputs with mixed or weak keyword signals, the raw SVM prediction is used without correction.

Tesseract OCR accuracy is dependent on image quality. Blurry, low-contrast, or stylized text will produce degraded results. Pre-processing (grayscale, denoise, threshold) improves accuracy on standard screenshots but does not handle all cases.

The Groq API call has a 15-second timeout. On slow connections or during API rate limiting, the explanation falls back to the rule-based output silently.

