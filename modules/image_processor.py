# ============================================================
# modules/image_processor.py  –  OCR via OpenCV + pytesseract
# ============================================================

import cv2
import pytesseract
import numpy as np
from PIL import Image


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better OCR results.
    Steps: grayscale → denoise → adaptive threshold
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding (binarise) – improves OCR on uneven backgrounds
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OpenCV pre-processing
    followed by pytesseract OCR.

    Parameters
    ----------
    image_path : str  – path to the image file

    Returns
    -------
    str  – extracted text (may be empty if OCR finds nothing)

    Raises
    ------
    FileNotFoundError if image_path does not exist.
    RuntimeError on OpenCV / pytesseract errors.
    """
    # Load with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        # Fallback: try via PIL (handles more formats)
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not open image at '{image_path}': {exc}"
            )

    # Pre-process for better OCR
    processed = preprocess_image(img)

    # OCR config: --psm 3 = fully automatic page segmentation
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(processed, config=custom_config)

    return text.strip()
