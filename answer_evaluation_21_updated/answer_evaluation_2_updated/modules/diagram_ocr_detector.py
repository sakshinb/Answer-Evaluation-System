"""
DiagramOCRDetector
------------------
Detects whether a student has submitted image/diagram uploads in their answer
file and routes them through OCR before scoring.

Capabilities
------------
1. PDF page rasterisation → image detection
2. Standalone image-file detection (PNG, JPG, etc.)
3. OCR via pytesseract (if installed) or pdf2image + pillow fallback
4. Returns extracted text so it can be appended to the student answer
   before NLP scoring, making diagram content scorable.

Usage
-----
    detector = DiagramOCRDetector()
    result   = detector.process_submission(file_path)
    # result['text']      – OCR'd text (empty str if no images)
    # result['has_images'] – bool
    # result['pages']     – list of per-page dicts
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _ocr_image(image) -> str:
    """
    Run OCR on a PIL Image object.
    Falls back to an empty string if pytesseract is unavailable.
    """
    try:
        import pytesseract
        return pytesseract.image_to_string(image, config='--psm 6').strip()
    except ImportError:
        logger.warning("pytesseract not installed – OCR skipped")
        return ""
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""


def _pdf_has_images(pdf_path: str) -> bool:
    """Quick check: does the PDF contain any XObject images?"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                resources = page.get('/Resources', {})
                if resources.get('/XObject'):
                    return True
        return False
    except Exception:
        return False


def _rasterise_pdf(pdf_path: str, dpi: int = 200) -> List:
    """
    Convert PDF pages to PIL Images using pdf2image.
    Returns empty list if pdf2image / poppler are unavailable.
    """
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi)
    except ImportError:
        logger.warning("pdf2image not installed – skipping PDF rasterisation")
        return []
    except Exception as e:
        logger.warning("PDF rasterisation failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Diagram classification heuristic
# ---------------------------------------------------------------------------

def _is_diagram_heavy(image) -> bool:
    """
    Simple heuristic: if a page image has very low text-pixel density
    (large blank or graphic regions) it is likely a diagram.
    Uses PIL pixel analysis – doesn't need pytesseract.
    """
    try:
        import numpy as np
        gray = image.convert('L')
        arr  = np.array(gray)
        # White pixel fraction (>240 out of 255)
        white_ratio = (arr > 240).sum() / arr.size
        # A nearly-all-white page is blank; a page with lots of midtone pixels
        # is likely a diagram/figure rather than text.
        midtone_ratio = ((arr > 60) & (arr < 200)).sum() / arr.size
        return midtone_ratio > 0.15 and white_ratio < 0.90
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DiagramOCRDetector:
    """
    Detects diagrams/images in a student submission and extracts their text.

    Parameters
    ----------
    ocr_diagrams : bool  – run OCR even on pages classified as diagram-heavy
    min_text_len : int   – minimum extracted text length to count as useful
    """

    def __init__(self, ocr_diagrams: bool = True, min_text_len: int = 20):
        self.ocr_diagrams = ocr_diagrams
        self.min_text_len = min_text_len

    def process_submission(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point.  Returns a dict with:
            has_images  : bool
            text        : str  (OCR'd text from image regions, or "")
            pages       : list[dict] with per-page info
            method      : str describing what was done
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'):
            return self._process_image_file(file_path)
        else:
            return {"has_images": False, "text": "", "pages": [], "method": "unsupported_format"}

    # ------------------------------------------------------------------

    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        pages_info: List[Dict] = []
        all_text: List[str]    = []

        has_img = _pdf_has_images(pdf_path)
        images  = _rasterise_pdf(pdf_path)

        if not images:
            return {"has_images": has_img, "text": "", "pages": [], "method": "pdf_no_rasterise"}

        for i, img in enumerate(images):
            is_diag = _is_diagram_heavy(img)
            ocr_text = ""
            if self.ocr_diagrams or not is_diag:
                ocr_text = _ocr_image(img)

            page_info = {
                "page": i + 1,
                "is_diagram": is_diag,
                "ocr_text_len": len(ocr_text),
                "ocr_text_preview": ocr_text[:120] if ocr_text else "",
            }
            pages_info.append(page_info)
            if ocr_text and len(ocr_text) >= self.min_text_len:
                all_text.append(ocr_text)

        combined = "\n\n".join(all_text)
        return {
            "has_images": has_img or any(p["is_diagram"] for p in pages_info),
            "text":       combined,
            "pages":      pages_info,
            "method":     "pdf_rasterise_ocr",
        }

    def _process_image_file(self, image_path: str) -> Dict[str, Any]:
        try:
            from PIL import Image
            img      = Image.open(image_path)
            is_diag  = _is_diagram_heavy(img)
            ocr_text = _ocr_image(img) if (self.ocr_diagrams or not is_diag) else ""
            return {
                "has_images": True,
                "text":       ocr_text,
                "pages":      [{"page": 1, "is_diagram": is_diag,
                                "ocr_text_len": len(ocr_text),
                                "ocr_text_preview": ocr_text[:120]}],
                "method":     "image_ocr",
            }
        except ImportError:
            logger.warning("Pillow not installed – image OCR skipped")
            return {"has_images": True, "text": "", "pages": [], "method": "image_no_pillow"}
        except Exception as e:
            logger.error("Image OCR failed: %s", e)
            return {"has_images": True, "text": "", "pages": [], "method": "image_error"}

    def enrich_student_text(self, student_text: str, file_path: str) -> Dict[str, Any]:
        """
        Convenience wrapper: run OCR on file_path and append any extracted
        text to student_text.  Returns dict with 'enriched_text' and detection info.
        """
        result = self.process_submission(file_path)
        enriched = student_text
        if result["text"]:
            enriched = student_text + "\n\n[Diagram/Image content (OCR)]\n" + result["text"]
        result["enriched_text"]    = enriched
        result["ocr_text_appended"] = bool(result["text"])
        return result