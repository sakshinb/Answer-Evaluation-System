"""
trocr_ocr.py
------------
Replaces the old PyPDF2/pdfplumber text extraction with a proper OCR pipeline
using microsoft/trocr-base-handwritten for handwritten answer sheets.

Pipeline
--------
For PDF input:
  1. Convert every PDF page → high-res PIL image  (pdf2image / fitz fallback)
  2. Run each image through the TrOCR pipeline
  3. Concatenate all page texts and run post-processing

For image input (jpg/png/jpeg/webp):
  1. Load image directly with PIL
  2. Run TrOCR
  3. Post-process

Post-processing
  - Remove TrOCR hallucination artifacts (repeated chars, lone symbols)
  - Normalise whitespace
  - Merge short lines that belong to the same sentence

The class is a singleton-friendly object: load the model once, call extract() many times.
"""

import re
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID   = "microsoft/trocr-base-handwritten"
CHUNK_SIZE = 384          # pixels — TrOCR accepts any width but works best ≤ 384 tall strips
DPI        = 300          # PDF render resolution; higher = better OCR, slower


# ── Lazy imports (heavy deps loaded only when first used) ──────────────────────

def _load_pdf_renderer():
    """Return a function  pdf_path -> List[PIL.Image]"""
    # Prefer pdf2image (uses poppler)
    try:
        from pdf2image import convert_from_path
        def render(path: str) -> list:
            return convert_from_path(path, dpi=DPI)
        logger.info("TrOCR OCR: using pdf2image for PDF rendering")
        return render
    except ImportError:
        pass

    # Fallback: PyMuPDF (fitz)
    try:
        import fitz
        from PIL import Image
        import io
        def render(path: str) -> list:
            doc = fitz.open(path)
            images = []
            for page in doc:
                mat = fitz.Matrix(DPI / 72, DPI / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
            return images
        logger.info("TrOCR OCR: using PyMuPDF (fitz) for PDF rendering")
        return render
    except ImportError:
        pass

    raise ImportError(
        "Cannot render PDF pages. Install either:\n"
        "  pip install pdf2image          (needs poppler)\n"
        "  pip install PyMuPDF            (self-contained)"
    )


def _slice_image_into_strips(image, strip_height: int = CHUNK_SIZE) -> list:
    """
    Slice a tall page image into horizontal strips so TrOCR sees line-sized chunks.
    TrOCR was trained on single-line images; feeding a full page degrades accuracy.
    """
    from PIL import Image
    w, h = image.size
    strips = []
    y = 0
    while y < h:
        box = (0, y, w, min(y + strip_height, h))
        strip = image.crop(box)
        # Skip nearly-blank strips (white / empty)
        grayscale = strip.convert("L")
        pixels = list(grayscale.getdata())
        if pixels:
            mean_brightness = sum(pixels) / len(pixels)
            if mean_brightness < 250:          # not a blank strip
                strips.append(strip)
        y += strip_height
    return strips


class TrOCRExtractor:
    """
    Extracts text from handwritten images/PDFs using microsoft/trocr-base-handwritten.

    Usage:
        extractor = TrOCRExtractor()              # loads model on first call
        text = extractor.extract_from_pdf("sheet.pdf")
        text = extractor.extract_from_image("page.jpg")
        text = extractor.extract("anything.pdf")  # auto-detects type
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto"):
        self._model_id    = model_id
        self._device      = device
        self._processor   = None
        self._model       = None
        self._pdf_render  = None

    # ── Lazy model loader ──────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._processor is not None:
            return

        logger.info("TrOCR OCR: loading model %s (first-time, may take ~30 s)...", self._model_id)
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            if self._device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device

            self._processor = TrOCRProcessor.from_pretrained(self._model_id)
            self._model     = VisionEncoderDecoderModel.from_pretrained(self._model_id).to(device)
            self._device_str = device
            logger.info("TrOCR OCR: model loaded on %s", device)
        except Exception as e:
            logger.error("TrOCR OCR: failed to load model — %s", e)
            raise

    def _ensure_pdf_renderer(self):
        if self._pdf_render is None:
            self._pdf_render = _load_pdf_renderer()

    # ── Core OCR ──────────────────────────────────────────────────────────────

    def _ocr_strip(self, strip) -> str:
        """Run TrOCR on a single PIL image strip and return the predicted text."""
        from PIL import Image
        import torch

        # TrOCR needs RGB
        strip_rgb = strip.convert("RGB")

        pixel_values = self._processor(
            images=strip_rgb,
            return_tensors="pt"
        ).pixel_values.to(self._device_str)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _ocr_full_image(self, image) -> str:
        """Slice a full-page image into strips, OCR each, and reassemble."""
        self._ensure_loaded()
        strips = _slice_image_into_strips(image)
        if not strips:
            return ""

        lines = []
        for strip in strips:
            try:
                line = self._ocr_strip(strip)
                if line:
                    lines.append(line)
            except Exception as e:
                logger.warning("TrOCR OCR: strip failed — %s", e)
                continue

        return "\n".join(lines)

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_from_image(self, image_path: str) -> str:
        """Extract text from a single handwritten image file."""
        from PIL import Image
        self._ensure_loaded()
        img = Image.open(image_path)
        raw = self._ocr_full_image(img)
        return self._postprocess(raw)

    def extract_from_image_object(self, pil_image) -> str:
        """Extract text from an already-open PIL Image object."""
        self._ensure_loaded()
        raw = self._ocr_full_image(pil_image)
        return self._postprocess(raw)

    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF containing handwritten answer sheets.
        Each page is rendered at DPI=300 then OCR'd line-by-line.
        """
        self._ensure_loaded()
        self._ensure_pdf_renderer()

        try:
            pages = self._pdf_render(pdf_path)
        except Exception as e:
            logger.error("TrOCR OCR: PDF render failed for %s — %s", pdf_path, e)
            return self._fallback_pdf_extract(pdf_path)

        all_text = []
        for page_num, page_img in enumerate(pages, 1):
            logger.info("TrOCR OCR: processing page %d/%d", page_num, len(pages))
            try:
                page_text = self._ocr_full_image(page_img)
                if page_text.strip():
                    all_text.append(f"[Page {page_num}]\n{page_text}")
            except Exception as e:
                logger.warning("TrOCR OCR: page %d failed — %s", page_num, e)
                continue

        raw = "\n\n".join(all_text)
        return self._postprocess(raw)

    def extract(self, file_path: str) -> str:
        """
        Auto-detect file type and extract text.
        Supports: .pdf, .jpg, .jpeg, .png, .webp, .bmp, .tiff
        """
        ext = Path(file_path).suffix.lower().lstrip(".")
        if ext == "pdf":
            return self.extract_from_pdf(file_path)
        elif ext in {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"}:
            return self.extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type for TrOCR: .{ext}")

    # ── Post-processing ────────────────────────────────────────────────────────

    def _postprocess(self, text: str) -> str:
        """Clean up TrOCR output: remove artifacts, normalise whitespace."""
        if not text:
            return ""

        # Remove page markers inserted above
        text = re.sub(r"\[Page \d+\]\n?", "", text)

        # TrOCR sometimes repeats characters (e.g. "tttthe")
        text = re.sub(r"(.)\1{4,}", r"\1\1", text)

        # Remove lines that are purely punctuation/symbols (hallucinations)
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Keep line only if it has at least 2 alphanumeric characters
            if len(re.findall(r"[a-zA-Z0-9]", stripped)) >= 2:
                clean_lines.append(stripped)

        # Merge lines that appear to be mid-sentence breaks
        merged = []
        buffer = ""
        for line in clean_lines:
            if buffer and not buffer.endswith((".", "?", "!", ":")):
                buffer += " " + line
            else:
                if buffer:
                    merged.append(buffer)
                buffer = line
        if buffer:
            merged.append(buffer)

        text = "\n".join(merged)

        # Normalise multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ── Fallback ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_pdf_extract(pdf_path: str) -> str:
        """
        Last-resort text extraction using PyPDF2 / pdfplumber.
        Used only when PDF rendering fails entirely.
        """
        logger.warning("TrOCR OCR: falling back to PyPDF2/pdfplumber for %s", pdf_path)
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            pass
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            pass
        return ""


# ── Module-level singleton ─────────────────────────────────────────────────────

_extractor: Optional[TrOCRExtractor] = None

def get_extractor() -> TrOCRExtractor:
    """Return the shared TrOCRExtractor instance (model loaded once)."""
    global _extractor
    if _extractor is None:
        _extractor = TrOCRExtractor()
    return _extractor


def extract_text(file_path: str) -> str:
    """
    Convenience function — call this instead of extract_text_from_pdf() in app.py.
    Works for PDF and image files.
    """
    return get_extractor().extract(file_path)