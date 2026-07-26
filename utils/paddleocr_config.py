import os
from typing import Dict


def paddleocr_kwargs(lang: str = "en") -> Dict[str, object]:
    """Shared PaddleOCR defaults for this project."""
    kwargs: Dict[str, object] = {
        "lang": lang,
        "ocr_version": os.environ.get("PADDLEOCR_VERSION", "PP-OCRv6"),
        "text_detection_model_name": os.environ.get(
            "PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
        ),
        "text_recognition_model_name": os.environ.get(
            "PADDLEOCR_REC_MODEL", "PP-OCRv6_medium_rec"
        ),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }

    det_model_dir = os.environ.get("PADDLEOCR_DET_MODEL_DIR")
    rec_model_dir = os.environ.get("PADDLEOCR_REC_MODEL_DIR")
    if det_model_dir:
        kwargs["text_detection_model_dir"] = det_model_dir
        kwargs.pop("text_detection_model_name", None)
    if rec_model_dir:
        kwargs["text_recognition_model_dir"] = rec_model_dir
        kwargs.pop("text_recognition_model_name", None)

    return kwargs
