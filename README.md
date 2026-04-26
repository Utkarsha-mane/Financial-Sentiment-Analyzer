# Financial News Sentiment Analyzer
## Setup & Run Instructions

---

## 1. Prerequisites

- Python 3.9 or newer
- pip

---

## 2. Install Python dependencies

```bash
cd project
pip install -r requirements.txt
```

If you see a `scikit-learn` version warning you can ignore it — the model still works.

---

## 3. Install Tesseract OCR (for image input)

### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add that folder to your system PATH  
   **OR** add this line to `config/settings.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

### macOS
```bash
brew install tesseract
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr
```

---

## 4. Configure Groq API key (optional but recommended)

Set your Groq API key as an environment variable or use the local `.env` file:

```bash
# Linux / macOS
export GROQ_API_KEY="gsk_your_key_here"

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
```

Or add it to the project `.env` file instead:
```text
GROQ_API_KEY=gsk_your_key_here
```

If you use `.env`, it is loaded automatically by `config/settings.py`.

Get a free key at: https://console.groq.com

---

## 5. Place model files

Put your trained model files in the `model/` directory:

```
project/
└── model/
    ├── svm_model.pkl      ← trained LinearSVC
    ├── tfidf.pkl          ← TF-IDF vectorizer (parameters)
    └── keywords.json      ← positive/negative keyword lists
```

The `tfidf_fitted.pkl` is generated automatically on first run.

---

## 6. Run the application

```bash
cd project
python main.py
```

---

## Usage

### Text input
Paste any financial news text into the **Text** tab and click **Analyze Text**.

### URL input
Paste a news article URL into the **URL** tab and click **Fetch & Analyze URL**.
Requires internet access.

### Image input
Click **Browse Image…** in the **Image** tab to select a screenshot or photo
of a news article, then click **Analyze Image**.
Requires Tesseract OCR to be installed.

### Save results
After each analysis click **Save to Database** to persist the result.

### View charts
Click **📊 Show Charts / Dashboard** to open the analytics dashboard
(requires at least one saved result).

---

## Project structure

```
project/
├── main.py                        # entry point
├── requirements.txt
├── config/
│   └── settings.py               # API keys, file paths
├── model/
│   ├── svm_model.pkl
│   ├── tfidf.pkl
│   ├── tfidf_fitted.pkl          # auto-generated on first run
│   └── keywords.json
├── data/
│   └── processed_data.csv
├── modules/
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   ├── model_loader.py
│   ├── analyzer.py
│   ├── image_processor.py
│   ├── url_processor.py
│   ├── visualizer.py
│   ├── database_manager.py
│   └── explanation_generator.py
└── ui/
    └── app_gui.py
```
