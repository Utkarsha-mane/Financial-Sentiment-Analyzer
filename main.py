#!/usr/bin/env python3
# main.py  –  Application entry point

import os
import sys

# Ensure project root is always on the import path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.app_gui import FinancialSentimentApp


def main():
    app = FinancialSentimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
