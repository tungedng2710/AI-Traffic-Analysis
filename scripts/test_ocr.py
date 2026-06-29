import argparse
import re
import sys, os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.paddleocr_config import paddleocr_kwargs
from utils.utils import check_legit_plate

@lru_cache(maxsize=1)
def get_ocr(use_mobile: bool = False) -> PaddleOCR:
    """Lazily construct and cache a PaddleOCR instance.

    Returns:
        PaddleOCR: Configured OCR engine instance.
    """
    if use_mobile:
        kwargs = paddleocr_kwargs(lang="en")
        kwargs["device"] = "cpu"
        return PaddleOCR(**kwargs)
    else:
        return PaddleOCR(**paddleocr_kwargs(lang="en"))


def extract_plate_info(
    plate_image: np.ndarray,
    conf_thres: float = 0.5,
    ocr: Optional[PaddleOCR] = None,
) -> Tuple[str, float]:
    """Extract license plate text and confidence from an image.

    Args:
        plate_image: OpenCV image (BGR) containing a license plate.
        conf_thres: Minimum average confidence to accept the OCR result.
        ocr: Optional OCR engine; if not provided, uses a cached PaddleOCR.

    Returns:
        Tuple[str, float]: (plate_text, avg_confidence). Returns ('', 0.0) if
        confidence is below threshold or the plate text is not valid.
    """
    if plate_image is None or not isinstance(plate_image, np.ndarray):
        return "", 0.0

    engine = ocr or get_ocr(use_mobile=True)

    try:
        results = engine.predict(input=plate_image)
    except Exception:
        # If OCR fails for any reason, treat as no result
        return "", 0.0

    if not results:
        return "", 0.0

    rec_texts = results[0].get("rec_texts", [])
    rec_scores = results[0].get("rec_scores", [])

    plate_info = " ".join(rec_texts) if rec_texts else ""

    # Sanitize: keep only alphanumerics, dash, dot
    plate_info = re.sub(r"[^A-Za-z0-9\-.]", "", plate_info)

    # Heuristic: if starts with a letter and the 3rd char equals 'C', replace with '0'
    if plate_info and plate_info[0].isalpha() and len(plate_info) > 2:
        if plate_info[2] == "C":
            plate_info = plate_info[:2] + "0" + plate_info[3:]

    conf = float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0

    if conf >= conf_thres and check_legit_plate(plate_info):
        return plate_info, conf
    return "", 0.0


def _cli() -> None:
    parser = argparse.ArgumentParser(description="OCR license plate text from an image.")
    parser.add_argument("image", help="Path to input image (OpenCV-readable)")
    parser.add_argument("--thres", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    text, conf = extract_plate_info(img, conf_thres=args.thres)
    print({"text": text, "confidence": conf})


if __name__ == "__main__":
    _cli()
