# ui/app_gui.py  –  Tkinter desktop GUI

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Make sure the project root is on the path when launched directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.analyzer         import analyze
from modules.image_processor  import extract_text_from_image
from modules.url_processor    import extract_text_from_url
from modules.database_manager import init_db, save_result
from modules.visualizer       import show_dashboard
from config.settings          import GROQ_API_KEY


# Colour / font constants 
BG_DARK     = "#1a1a2e"
BG_CARD     = "#16213e"
BG_INPUT    = "#0f3460"
ACCENT_BLUE = "#533483"
TEXT_LIGHT  = "#e0e0e0"
TEXT_DIM    = "#a0a0b0"

SENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#3498db",
}

FONT_TITLE  = ("Helvetica", 20, "bold")
FONT_HEADER = ("Helvetica", 12, "bold")
FONT_BODY   = ("Helvetica", 10)
FONT_MONO   = ("Courier", 10)


class FinancialSentimentApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # DB init 
        init_db()

        # Window setup 
        self.title("Financial News Sentiment Analyzer")
        self.geometry("1050x780")
        self.minsize(800, 600)
        self.configure(bg=BG_DARK)
        self._center_window()

        # State
        self._use_groq = tk.BooleanVar(value=bool(GROQ_API_KEY))
        self._last_result: dict = {}
        self._analyzing = False   # busy flag — prevents double-clicks

        self._build_ui()

    # Layout helpers 

    def _center_window(self):
        self.update_idletasks()
        w, h = 1050, 780
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _card(self, parent, **kw) -> tk.Frame:
        """Styled frame acting as a 'card'."""
        return tk.Frame(parent, bg=BG_CARD, bd=0, **kw)

    # Build UI 

    def _build_ui(self):
        # Title bar 
        title_bar = tk.Frame(self, bg=ACCENT_BLUE, height=60)
        title_bar.pack(fill="x")
        tk.Label(
            title_bar,
            text="📈  Financial News Sentiment Analyzer",
            font=FONT_TITLE, fg="white", bg=ACCENT_BLUE,
        ).pack(side="left", padx=20, pady=10)

        # Groq toggle on the right
        # groq_frame = tk.Frame(title_bar, bg=ACCENT_BLUE)
        # groq_frame.pack(side="right", padx=20)
        # ttk.Checkbutton(
        #     groq_frame, text="Use AI (Groq)",
        #     variable=self._use_groq, style="TCheckbutton",
        # ).pack(side="left")

        # Main paned layout 
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # Left column  (inputs)
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        # Right column (results)
        right = tk.Frame(main, bg=BG_DARK, width=380)
        right.pack(side="right", fill="both", padx=(8, 0))
        right.pack_propagate(False)

        self._build_input_section(left)
        self._build_result_section(right)

    # Left – Input section 

    def _build_input_section(self, parent):

        # Tab control: Text / URL / Image 
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.TNotebook",
                         background=BG_DARK, borderwidth=0)
        style.configure("Custom.TNotebook.Tab",
                         background=BG_CARD, foreground=TEXT_LIGHT,
                         padding=[12, 6], font=FONT_HEADER)
        style.map("Custom.TNotebook.Tab",
                  background=[("selected", ACCENT_BLUE)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(parent, style="Custom.TNotebook")
        nb.pack(fill="both", expand=True)

        # Tab 1: Text 
        t1 = self._card(nb)
        nb.add(t1, text="  ✍  Text  ")

        tk.Label(t1, text="Paste financial news text below:",
                 font=FONT_HEADER, fg=TEXT_LIGHT, bg=BG_CARD).pack(
                     anchor="w", padx=12, pady=(10, 2))

        self.text_input = scrolledtext.ScrolledText(
            t1, height=14, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, insertbackground="white",
            relief="flat", wrap="word",
        )
        self.text_input.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        btn_row = tk.Frame(t1, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        self._styled_button(btn_row, "  Analyze Text  ",
                            self._on_analyze_text).pack(side="left")
        self._styled_button(btn_row, " Clear ",
                            lambda: self.text_input.delete("1.0", "end"),
                            secondary=True).pack(side="left", padx=(8, 0))

        # Tab 2: URL 
        t2 = self._card(nb)
        nb.add(t2, text="  🔗  URL  ")

        tk.Label(t2, text="Enter a financial news article URL:",
                 font=FONT_HEADER, fg=TEXT_LIGHT, bg=BG_CARD).pack(
                     anchor="w", padx=12, pady=(10, 2))

        url_row = tk.Frame(t2, bg=BG_CARD)
        url_row.pack(fill="x", padx=12, pady=(0, 6))

        self.url_input = tk.Entry(
            url_row, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, insertbackground="white",
            relief="flat", bd=6,
        )
        self.url_input.pack(fill="x")
        self.url_input.insert(0, "https://")

        # Extracted preview
        tk.Label(t2, text="Extracted article preview:",
                font=FONT_HEADER, fg=TEXT_DIM, bg=BG_CARD).pack(
                    anchor="w", padx=12, pady=(6, 2))
        self.url_preview = scrolledtext.ScrolledText(
            t2, height=10, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, relief="flat",
            state="disabled", wrap="word",
        )
        self.url_preview.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self._styled_button(t2, "  Fetch & Analyze URL  ",
                            self._on_analyze_url).pack(
                                anchor="w", padx=12, pady=(0, 10))

        # Tab 3: Image 
        t3 = self._card(nb)
        nb.add(t3, text="  🖼  Image  ")

        tk.Label(t3, text="Upload a screenshot or image of a news article:",
                 font=FONT_HEADER, fg=TEXT_LIGHT, bg=BG_CARD).pack(
                     anchor="w", padx=12, pady=(10, 2))

        self.img_path_var = tk.StringVar(value="No image selected")
        tk.Label(t3, textvariable=self.img_path_var,
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_CARD,
                 wraplength=500).pack(anchor="w", padx=12, pady=(0, 6))

        btn_row2 = tk.Frame(t3, bg=BG_CARD)
        btn_row2.pack(fill="x", padx=12)
        self._styled_button(btn_row2, "  Browse Image…  ",
                            self._on_browse_image).pack(side="left")
        self._styled_button(btn_row2, "  Analyze Image  ",
                            self._on_analyze_image).pack(side="left", padx=(8, 0))

        # OCR text preview
        tk.Label(t3, text="OCR extracted text:",
                font=FONT_HEADER, fg=TEXT_DIM, bg=BG_CARD).pack(
                    anchor="w", padx=12, pady=(10, 2))
        self.ocr_preview = scrolledtext.ScrolledText(
            t3, height=10, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, relief="flat",
            state="disabled", wrap="word",
        )
        self.ocr_preview.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # Bottom action buttons (always visible below tabs) 
        bottom_row = tk.Frame(parent, bg=BG_DARK)
        bottom_row.pack(fill="x", padx=4, pady=(10, 4))

        self._styled_button(
            bottom_row, "  📊 Show Dashboard  ",
            self._on_show_charts,
        ).pack(side="left")

        # Save button — green, always visible
        self.save_btn = tk.Button(
            bottom_row,
            text="  💾 Save to Database  ",
            command=self._on_save,
            font=FONT_HEADER,
            fg="white",
            bg="#27ae60",
            activeforeground="white",
            activebackground="#1e8449",
            disabledforeground="#95a5a6",
            relief="flat", bd=0,
            padx=12, pady=6,
            cursor="hand2",
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=(10, 0))

    # Right – Results section 

    def _build_result_section(self, parent):
        tk.Label(parent, text="Analysis Result",
                 font=FONT_TITLE, fg=TEXT_LIGHT, bg=BG_DARK).pack(
                     anchor="w", pady=(4, 6))

        # Sentiment badge 
        badge_frame = self._card(parent)
        badge_frame.pack(fill="x", pady=(0, 6))

        self.badge_label = tk.Label(
            badge_frame,
            text="⬤  —",
            font=("Helvetica", 22, "bold"),
            fg=TEXT_DIM, bg=BG_CARD,
        )
        self.badge_label.pack(pady=12)

        self.fin_label = tk.Label(
            badge_frame,
            text="",
            font=FONT_BODY, fg=TEXT_DIM, bg=BG_CARD,
        )
        self.fin_label.pack(pady=(0, 8))

        # Score card 
        score_card = self._card(parent)
        score_card.pack(fill="x", pady=(0, 6))

        score_titles = ["Positive Score", "Negative Score",
                        "Keyword Strength", "Sentiment Ratio"]
        self._score_vars = []
        for i, title in enumerate(score_titles):
            row = tk.Frame(score_card, bg=BG_CARD)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=title + ":", font=FONT_BODY,
                     fg=TEXT_DIM, bg=BG_CARD, width=18,
                     anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            self._score_vars.append(var)
            tk.Label(row, textvariable=var, font=(FONT_HEADER[0], 10, "bold"),
                     fg=TEXT_LIGHT, bg=BG_CARD).pack(side="left")

        # Keywords 
        kw_card = self._card(parent)
        kw_card.pack(fill="x", pady=(0, 6))
        tk.Label(kw_card, text="Matched Keywords",
                 font=FONT_HEADER, fg=TEXT_LIGHT, bg=BG_CARD).pack(
                     anchor="w", padx=12, pady=(8, 2))
        self.kw_text = tk.Text(
            kw_card, height=4, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, relief="flat",
            state="disabled", wrap="word",
        )
        self.kw_text.pack(fill="x", padx=12, pady=(0, 8))

        # Explanation
        exp_card = self._card(parent)
        exp_card.pack(fill="both", expand=True)
        tk.Label(exp_card, text="Explanation",
                 font=FONT_HEADER, fg=TEXT_LIGHT, bg=BG_CARD).pack(
                     anchor="w", padx=12, pady=(8, 2))
        self.exp_text = scrolledtext.ScrolledText(
            exp_card, font=FONT_BODY,
            bg=BG_INPUT, fg=TEXT_LIGHT, relief="flat",
            state="disabled", wrap="word",
        )
        self.exp_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # Status bar 
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(parent, textvariable=self.status_var,
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_DARK).pack(
                     anchor="w", padx=4)

    # Widget factory 

    def _styled_button(self, parent, text: str,
                       command, secondary=False) -> tk.Button:
        bg  = BG_INPUT   if secondary else ACCENT_BLUE
        fg  = TEXT_DIM   if secondary else "white"
        abg = "#2c2f6b"  if secondary else "#6a45a0"
        btn = tk.Button(
            parent, text=text, command=command,
            font=FONT_HEADER, fg=fg, bg=bg,
            activeforeground="white", activebackground=abg,
            relief="flat", bd=0, padx=10, pady=6, cursor="hand2",
        )
        return btn

    # Handlers 

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _run_in_thread(self, fn):
        """Run fn in a background thread to keep UI responsive."""
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    # Analyze text 

    def _on_analyze_text(self):
        if self._analyzing:
            messagebox.showinfo("Busy", "Analysis in progress, please wait…")
            return
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some text to analyze.")
            return
        self._set_status("Analyzing …")
        self._run_in_thread(lambda: self._do_analyze(text, "text"))

    # Analyze URL 

    def _on_analyze_url(self):
        if self._analyzing:
            messagebox.showinfo("Busy", "Analysis in progress, please wait…")
            return
        url = self.url_input.get().strip()
        if not url or url == "https://":
            messagebox.showwarning("Empty URL", "Please enter a valid URL.")
            return
        self._set_status("Fetching article …")
        self._run_in_thread(lambda: self._do_analyze_url(url))

    def _do_analyze_url(self, url: str):
        try:
            text, financial = extract_text_from_url(url)
            if not text:
                self.after(0, lambda: messagebox.showerror(
                    "Extraction Failed",
                    "Could not extract text from that URL.\n"
                    "Check the URL and your internet connection."
                ))
                self._set_status("URL extraction failed.")
                return
            # Show preview
            self.after(0, lambda: self._update_preview(self.url_preview, text[:800]))
            self._do_analyze(text, "url")
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self._set_status("Error.")

    # Browse image 

    def _on_browse_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if path:
            self.img_path_var.set(path)
            self._update_preview(self.ocr_preview, "")

    def _on_analyze_image(self):
        if self._analyzing:
            messagebox.showinfo("Busy", "Analysis in progress, please wait…")
            return
        path = self.img_path_var.get()
        if not path or path == "No image selected":
            messagebox.showwarning("No Image", "Please browse and select an image first.")
            return
        self._set_status("Running OCR …")
        self._run_in_thread(lambda: self._do_analyze_image(path))

    def _do_analyze_image(self, path: str):
        try:
            text = extract_text_from_image(path)
            if not text:
                self.after(0, lambda: messagebox.showerror(
                    "OCR Failed",
                    "Could not extract text from the image.\n"
                    "Try a clearer image with readable text."
                ))
                self._set_status("OCR failed.")
                return
            self.after(0, lambda: self._update_preview(self.ocr_preview, text))
            self._do_analyze(text, "image")
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self._set_status("Error.")

    # Core analysis 

    def _do_analyze(self, text: str, source_type: str):
        self._analyzing = True
        self.after(0, lambda: self._set_status("Analyzing … please wait"))
        try:
            result = analyze(text, use_groq=self._use_groq.get())
            result["_source_type"] = source_type
            result["_raw_text"]    = text
            self._last_result = result
            self.after(0, lambda: self._display_result(result))
        except Exception as exc:
            import traceback
            err_msg = f"{type(exc).__name__}: {exc}"
            print("[ERROR]", traceback.format_exc())   # print to terminal
            self.after(0, lambda: messagebox.showerror("Analysis Error", err_msg))
            self.after(0, lambda: self._set_status(f"Error: {err_msg[:80]}"))
        finally:
            self._analyzing = False

    # Display result 

    def _display_result(self, result: dict):
        sentiment = result.get("sentiment", "neutral")
        color     = SENT_COLORS.get(sentiment, "#aaaaaa")
        icon      = {"positive": "📈", "negative": "📉", "neutral": "📊"}.get(sentiment, "")

        # Badge
        self.badge_label.config(
            text=f"{icon}  {sentiment.upper()}",
            fg=color,
        )

        # Financial tag
        is_fin = result.get("is_financial", False)
        self.fin_label.config(
            text="✅ Financial Content Detected" if is_fin else "⚠️ May not be financial news",
            fg="#2ecc71" if is_fin else "#e67e22",
        )

        # Scores
        vals = [
            str(result.get("pos_score",        "—")),
            str(result.get("neg_score",         "—")),
            str(result.get("keyword_strength",  "—")),
            f"{result.get('sentiment_ratio', 0):.2f}",
        ]
        for var, val in zip(self._score_vars, vals):
            var.set(val)

        # Keywords
        pos_kw = result.get("matched_positive", [])
        neg_kw = result.get("matched_negative", [])
        kw_str = ""
        if pos_kw:
            kw_str += f"Positive: {', '.join(pos_kw[:8])}\n"
        if neg_kw:
            kw_str += f"Negative: {', '.join(neg_kw[:8])}"
        if not kw_str:
            kw_str = "No strong keywords matched."
        self._set_text_widget(self.kw_text, kw_str)

        # Explanation
        exp = result.get("combined_explanation", result.get("rule_explanation", ""))
        self._set_text_widget(self.exp_text, exp)

        # Enable save
        self.save_btn.config(state="normal")
        self._set_status(f"Done. Sentiment: {sentiment.upper()}")

    # Save 

    def _on_save(self):
        if not self._last_result:
            return
        save_result(
            self._last_result,
            source_type = self._last_result.get("_source_type", "text"),
            raw_text    = self._last_result.get("_raw_text",    ""),
        )
        messagebox.showinfo("Saved", "Result saved to database successfully.")
        self.save_btn.config(state="disabled")
        self._set_status("Saved to database.")

    # Dashboard 

    def _on_show_charts(self):
        self._run_in_thread(show_dashboard)

    # Helpers 

    def _set_text_widget(self, widget, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _update_preview(self, widget, text: str):
        self._set_text_widget(widget, text)