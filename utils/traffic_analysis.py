from typing import Dict, List, Optional
import re
import inspect

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

from tracking.deep_sort import DeepSort
from tracking.sort import Sort
from utils.utils import (
    BGR_COLORS,
    VEHICLES,   
    Vehicle,
    check_image_size,
    check_legit_plate,
    compute_color,
    crop_expanded_plate,
    draw_text,
    gettime,
    map_label,
)


class TrafficAnalysisService:
    """
    AI Traffic Analysis Service - Core processing pipeline.
    
    Analyzes traffic scenes with vehicle detection, license plate recognition,
    and multi-object tracking capabilities.

    - Detects vehicles and plates with YOLO
    - Tracks with SORT/DeepSORT
    - OCR with PaddleOCR
    - Exposes `process_frame` and `process_image`
    """

    def __init__(self, opts):
        self.opts = opts

        requested_device = getattr(self.opts, "device", "auto")
        self.opts.device = self._resolve_device(requested_device)
        self._is_cuda = str(self.opts.device).lower().startswith("cuda")

        # Detectors
        self.vehicle_detector = self._create_vehicle_detector(self.opts.vehicle_weight)
        self.plate_detector = YOLO(self.opts.plate_weight, task="detect")
        try:
            self.plate_detector.to(self.opts.device)
        except Exception:
            pass

        # OCR
        self.read_plate = bool(getattr(self.opts, "read_plate", True))
        ocr_kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        try:
            if "use_gpu" in inspect.signature(PaddleOCR.__init__).parameters:
                ocr_kwargs["use_gpu"] = self._is_cuda
        except (ValueError, TypeError):
            pass
        self.ocr = PaddleOCR(**ocr_kwargs)
        self.ocr_thres: float = float(getattr(self.opts, "ocr_thres", 0.9))

        # Tracking
        self.deepsort: bool = bool(getattr(self.opts, "deepsort", False))
        self.dsort_weight: str = str(getattr(self.opts, "dsort_weight", "weights/tracking/deepsort/ckpt.t7"))
        self.vehicles: Dict[int, Vehicle] = {}
        self._init_tracker()

        # Misc
        self.color = BGR_COLORS
        self.lang = getattr(self.opts, "lang", "en")

    def _create_vehicle_detector(self, weight_path: str) -> YOLO:
        detector = YOLO(weight_path, task="detect")
        try:
            detector.to(self.opts.device)
        except Exception:
            pass
        self.opts.vehicle_weight = weight_path
        return detector

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested is None:
            requested = "auto"
        requested = str(requested).strip().lower()
        if requested in {"auto", ""}:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.isdigit():
            return f"cuda:{requested}" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _init_tracker(self) -> None:
        if self.deepsort:
            self.tracker = DeepSort(
                self.dsort_weight,
                max_dist=0.2,
                min_confidence=0.3,
                nms_max_overlap=0.5,
                max_iou_distance=0.7,
                max_age=70,
                n_init=3,
                nn_budget=100,
                use_cuda=self._is_cuda,
            )
        else:
            self.tracker = Sort()
        self.vehicles = {}

    def reset(self) -> None:
        self._init_tracker()

    def set_vehicle_weight(self, weight_path: str) -> None:
        if not weight_path:
            raise ValueError("Vehicle weight path must be provided")
        self.vehicle_detector = self._create_vehicle_detector(weight_path)
        # reset tracker state so IDs restart for the new model
        self.reset()

    def _extract_plate_text(self, plate_image):
        results = self.ocr.predict(input=plate_image)
        if len(results) > 0:
            plate_info = " ".join(results[0].get("rec_texts", []))
            rec_scores = results[0].get("rec_scores", [])
            conf_val = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
            plate_info = re.sub(r"[^A-Za-z0-9\-.]", "", plate_info)
            if plate_info and len(plate_info) > 2 and plate_info[0].isalpha() and plate_info[2] == 'C':
                plate_info = plate_info[:2] + '0' + plate_info[3:]
            return plate_info, conf_val
        else:
            return "", 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        displayed_frame = frame.copy()
        t0 = gettime()

        # Vehicle detection
        detection = self.vehicle_detector(
            frame,
            verbose=False,
            imgsz=640,
            device=self.opts.device,
            conf=self.opts.vconf,
        )[0]
        boxes = detection.boxes

        det_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
        det_cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.empty((0,), dtype=int)

        label_lookup = VEHICLES.get(self.lang, VEHICLES.get("en", []))

        def resolve_label(cls_idx: int) -> str:
            try:
                return map_label(cls_idx, label_lookup)
            except Exception:
                return str(cls_idx)

        # Tracking
        try:
            if self.deepsort:
                outputs = self.tracker.update(boxes.cpu().xywh, boxes.cpu().conf, frame)
            else:
                outputs = self.tracker.update(boxes.cpu().xyxy).astype(int)
        except Exception:
            outputs = np.empty((0, 5), dtype=int)

        def _iou(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter <= 0:
                return 0.0
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - inter
            return float(inter) / float(union + 1e-6)

        in_frame_ids: List[int] = []
        det_label_for_track: Dict[int, str] = {}

        for i in range(len(outputs)):
            tid = int(outputs[i, -1])
            in_frame_ids.append(tid)
            if tid not in self.vehicles:
                self.vehicles[tid] = Vehicle(track_id=tid)
            v = self.vehicles[tid]
            v.bbox_xyxy = outputs[i, :4]

            x1, y1, x2, y2 = v.bbox_xyxy
            cv2.rectangle(
                displayed_frame, (int(x1), int(y1)), (int(x2), int(y2)), compute_color(tid), 1
            )

            if det_xyxy.shape[0] > 0:
                tb = np.array([float(x1), float(y1), float(x2), float(y2)])
                best_iou, best_idx = 0.0, None
                for j in range(det_xyxy.shape[0]):
                    iou_val = _iou(tb, det_xyxy[j])
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, j
                if best_idx is not None and best_iou > 0.1:
                    label_text = resolve_label(int(det_cls[best_idx]))
                    det_label_for_track[tid] = label_text
                    v.vehicle_type = label_text

        if det_xyxy.shape[0] > 0 and det_cls.size > 0:
            for idx in range(det_xyxy.shape[0]):
                box = det_xyxy[idx].astype(int)
                label_text = resolve_label(int(det_cls[idx]))
                try:
                    draw_text(
                        img=displayed_frame,
                        text=str(label_text),
                        pos=(int(box[0]), int(box[1])),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                except Exception:
                    continue

        for tid in in_frame_ids:
            if tid not in det_label_for_track:
                v = self.vehicles[tid]
                label_text = v.vehicle_type if v.vehicle_type else f"ID {tid}"
                x1, y1, x2, _ = v.bbox_xyxy.astype(int)
                try:
                    draw_text(
                        img=displayed_frame,
                        text=str(label_text),
                        pos=(int(x1), int(y1)),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                except Exception:
                    continue
        # Plate recognition
        if self.read_plate and in_frame_ids:
            active: List[Vehicle] = []
            crops: List[np.ndarray] = []
            for tid in in_frame_ids:
                v = self.vehicles[tid]
                box = v.bbox_xyxy.astype(int)
                success = (
                    (v.ocr_conf > self.ocr_thres)
                    and len(v.plate_number) > 5
                    and check_legit_plate(v.plate_number)
                )
                if success:
                    draw_text(
                        img=displayed_frame,
                        text=v.plate_number,
                        pos=(box[0], box[1] + 26),
                        text_color=self.color["blue"],
                        text_color_bg=self.color["green"],
                    )
                    continue

                crop = frame[box[1] : box[3], box[0] : box[2], :]
                v.vehicle_image = crop
                if not check_image_size(crop, 112, 112):
                    continue
                active.append(v)
                crops.append(crop)

            if crops:
                detections = self.plate_detector(
                    crops,
                    verbose=False,
                    imgsz=320,
                    device=self.opts.device,
                    conf=self.opts.pconf,
                )
                with_plate: List[Vehicle] = []
                for idx, det in enumerate(detections):
                    v = active[idx]
                    crop = crops[idx]
                    box = v.bbox_xyxy.astype(int)
                    plate_xyxy = det.boxes.xyxy
                    if len(plate_xyxy) < 1:
                        continue
                    pxyxy = plate_xyxy[0].cpu().numpy().astype(int)
                    src = (int(pxyxy[0] + box[0]), int(pxyxy[1] + box[1]))
                    dst = (int(pxyxy[2] + box[0]), int(pxyxy[3] + box[1]))
                    cv2.rectangle(displayed_frame, src, dst, self.color["green"], 2)

                    try:
                        cropped_plate = crop_expanded_plate(pxyxy, crop, 0.15)
                    except Exception:
                        cropped_plate = np.zeros((8, 8, 3), dtype=np.uint8)

                    v.plate_image = cropped_plate
                    v.license_plate_bbox = pxyxy + np.array([box[0], box[1], box[0], box[1]])
                    with_plate.append(v)

                for v in with_plate:
                    text, conf = self._extract_plate_text(v.plate_image)
                    if conf > v.ocr_conf:
                        v.plate_number = text
                        v.ocr_conf = conf
                    if v.ocr_conf > self.ocr_thres and len(v.plate_number) > 5 and check_legit_plate(v.plate_number):
                        box = v.bbox_xyxy.astype(int)
                        draw_text(
                            img=displayed_frame,
                            text=v.plate_number,
                            pos=(box[0], box[1] + 26),
                            text_color=self.color["blue"],
                            text_color_bg=self.color["green"],
                        )

        # FPS overlay
        dt = gettime() - t0
        fps = int(round(1.0 / dt, 0)) if dt > 0 else 0
        draw_text(
            img=displayed_frame,
            text=f"FPS: {fps}",
            font_scale=1,
            font_thickness=2,
            text_color=self.color["blue"],
            text_color_bg=self.color["white"],
        )

        return displayed_frame

    def process_image(self, img: np.ndarray) -> np.ndarray:
        self.reset()
        return self.process_frame(img)
