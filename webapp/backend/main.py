import asyncio
import tempfile
import uuid
import re
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import cv2
import numpy as np
import json
import inspect
from types import SimpleNamespace
from typing import Optional, Set, Dict
from dataclasses import dataclass
from threading import Lock
from utils.traffic_analysis import TrafficAnalysisService
from utils.utils import BGR_COLORS, check_legit_plate, crop_expanded_plate, draw_text
import os
from typing import List
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from paddleocr import PaddleOCR
from ultralytics import YOLO

DEFAULT_DEVICE = os.environ.get("ALPR_DEVICE", "auto")

MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
UPLOAD_DIR = Path(tempfile.gettempdir()) / "alpr_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_VIDEO_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
uploaded_videos: Dict[str, Path] = {}
uploaded_videos_lock = Lock()

app = FastAPI()

# Allow cross-origin usage of the API (useful when serving frontend elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
REPO_ROOT = BASE_DIR.parent
CAMERA_CONFIG_PATH = REPO_ROOT / "webapp" / "rtsp_url.json"
VEHICLE_WEIGHTS_DIR = REPO_ROOT / "weights"
VEHICLE_WEIGHT_EXTENSIONS = {".pt", ".pth"}
PLATE_WEIGHT_PATH = REPO_ROOT / "weights" / "plate" / "license_plate_detector.pt"

# Initialize ALPR tracker (tracking + OCR) using shared core
opts = SimpleNamespace(
    vehicle_weight=str(REPO_ROOT / "weights" / "vehicle" / "vehicle_detector.pt"),
    plate_weight=str(REPO_ROOT / "weights" / "plate" / "license_plate_detector.pt"),
    dsort_weight=str(REPO_ROOT / "weights" / "tracking" / "deepsort" / "ckpt.t7"),
    vconf=0.6,
    pconf=0.25,
    ocr_thres=0.8,
    device=DEFAULT_DEVICE,
    deepsort=False,  # set True to use DeepSORT, else SORT
    read_plate=True,
    lang="en",  # follow main.py label mapping (car, bus, ...)
)
traffic_service = TrafficAnalysisService(opts)
ALPR_PROCESS_LOCK = Lock()
PARKING_PROCESS_LOCK = Lock()
peer_connections: Set[RTCPeerConnection] = set()


@dataclass
class ParkingPlateBBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ParkingDetection:
    bbox: ParkingPlateBBox
    score: float
    text: str
    text_conf: float
    legit: bool


class ParkingPlatePipeline:
    def __init__(self, weight_path: str, device: str, det_conf: float, ocr_conf: float) -> None:
        self.weight_path = Path(weight_path)
        self.device = traffic_service.opts.device if device == DEFAULT_DEVICE else device
        self.default_det_conf = det_conf
        self.default_ocr_conf = ocr_conf
        self.detector = YOLO(str(self.weight_path), task="detect")
        try:
            self.detector.to(self.device)
        except Exception:
            pass

        ocr_kwargs: Dict[str, object] = dict(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        try:
            if "use_gpu" in inspect.signature(PaddleOCR.__init__).parameters:
                ocr_kwargs["use_gpu"] = str(self.device).startswith("cuda")
        except (ValueError, TypeError):
            pass
        self.ocr = PaddleOCR(**ocr_kwargs)

    def _read_text(self, plate_image: np.ndarray) -> tuple[str, float, bool]:
        if plate_image is None or plate_image.size == 0:
            return "", 0.0, False
        results = self.ocr.predict(input=plate_image)
        if not results:
            return "", 0.0, False
        rec_texts = results[0].get("rec_texts", [])
        rec_scores = results[0].get("rec_scores", [])
        text = " ".join(rec_texts) if rec_texts else ""
        text = re.sub(r"[^A-Za-z0-9\-.]", "", text)
        if text and len(text) > 2 and text[0].isalpha() and text[2] == "C":
            text = text[:2] + "0" + text[3:]
        conf = float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
        return text, conf, bool(text and check_legit_plate(text))

    def __call__(self, image: np.ndarray, det_conf: Optional[float] = None, ocr_conf: Optional[float] = None) -> List[ParkingDetection]:
        det_conf = float(det_conf if det_conf is not None else self.default_det_conf)
        ocr_conf = float(ocr_conf if ocr_conf is not None else self.default_ocr_conf)
        pred = self.detector(image, verbose=False, conf=det_conf, imgsz=640, device=self.device)[0]
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        detections: List[ParkingDetection] = []
        for bbox_arr, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = bbox_arr.astype(int).tolist()
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            plate_crop = crop_expanded_plate((x1, y1, x2, y2), image, expand_ratio=0.25)
            text, text_conf, legit = self._read_text(plate_crop)
            if text_conf < ocr_conf:
                text = ""
                legit = False
            detections.append(
                ParkingDetection(
                    bbox=ParkingPlateBBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(score),
                    text=text,
                    text_conf=text_conf,
                    legit=legit,
                )
            )
        return detections


parking_service = ParkingPlatePipeline(
    weight_path=str(PLATE_WEIGHT_PATH),
    device=DEFAULT_DEVICE,
    det_conf=opts.pconf,
    ocr_conf=opts.ocr_thres,
)


def process_parking_frame(
    frame: np.ndarray,
    pconf: Optional[float] = None,
    ocr_conf: Optional[float] = None,
) -> np.ndarray:
    displayed_frame = frame.copy()
    detections = parking_service(
        frame,
        det_conf=pconf,
        ocr_conf=ocr_conf if ocr_conf is not None else opts.ocr_thres,
    )
    for detection in detections:
        bbox = detection.bbox
        cv2.rectangle(
            displayed_frame,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            BGR_COLORS["green"],
            2,
        )
        label = detection.text or "Plate"
        if detection.text_conf > 0:
            label = f"{label} {detection.text_conf:.2f}"
        draw_text(
            img=displayed_frame,
            text=label,
            pos=(bbox.x1, max(0, bbox.y1 - 4)),
            text_color=BGR_COLORS["blue"],
            text_color_bg=BGR_COLORS["green"],
        )
    return displayed_frame


def _find_vehicle_weights() -> List[Path]:
    if not VEHICLE_WEIGHTS_DIR.exists():
        return []
    weights: List[Path] = []
    for path in VEHICLE_WEIGHTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VEHICLE_WEIGHT_EXTENSIONS:
            continue
        if "vehicle" not in path.stem.lower():
            continue
        try:
            path.relative_to(VEHICLE_WEIGHTS_DIR)
        except ValueError:
            continue
        weights.append(path)
    weights.sort()
    return weights


def _serialize_weight_path(path: Path) -> dict:
    try:
        rel_path = path.relative_to(VEHICLE_WEIGHTS_DIR)
    except ValueError:
        rel_path = path
    label = path.stem.replace("_", " ").title()
    return {
        "label": label,
        "filename": path.name,
        "path": rel_path.as_posix(),
    }


def gen_frames(
    url: str,
    process: bool = False,
    vconf: Optional[float] = None,
    pconf: Optional[float] = None,
    read_plate: Optional[bool] = None,
):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if process:
            with ALPR_PROCESS_LOCK:
                if read_plate is not None:
                    traffic_service.read_plate = bool(read_plate)
                    setattr(traffic_service.opts, "read_plate", bool(read_plate))
                if vconf is not None:
                    traffic_service.opts.vconf = float(vconf)
                if pconf is not None:
                    traffic_service.opts.pconf = float(pconf)
                frame = traffic_service.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


def gen_parking_frames(
    url: str,
    pconf: Optional[float] = None,
    ocr_conf: Optional[float] = None,
):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with PARKING_PROCESS_LOCK:
            frame = process_parking_frame(frame, pconf=pconf, ocr_conf=ocr_conf)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


class ALPRWebRTCVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(
        self,
        *,
        url: str,
        process: bool,
        vconf: Optional[float],
        pconf: Optional[float],
        read_plate: Optional[bool],
        parking_mode: bool = False,
    ):
        super().__init__()
        self._url = url
        self._process = process
        self._parking_mode = parking_mode
        self._vconf = float(vconf) if vconf is not None else None
        self._pconf = float(pconf) if pconf is not None else None
        if read_plate is None:
            self._read_plate: Optional[bool] = None
        else:
            self._read_plate = bool(read_plate)
        self._cap = cv2.VideoCapture(url)
        if not self._cap or not self._cap.isOpened():
            self.close()
            raise ValueError("Unable to open video source")
        self._closed = False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    async def recv(self) -> av.VideoFrame:
        if self._closed:
            raise MediaStreamError("Video track already closed")

        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self._read_frame)

        if frame is None:
            self.close()
            raise MediaStreamError("Stream ended or failed to decode frame")

        if self._process:
            if self._parking_mode:
                with PARKING_PROCESS_LOCK:
                    frame = process_parking_frame(frame, pconf=self._pconf)
            else:
                with ALPR_PROCESS_LOCK:
                    if self._read_plate is not None:
                        traffic_service.read_plate = self._read_plate
                        setattr(traffic_service.opts, "read_plate", self._read_plate)
                    if self._vconf is not None:
                        traffic_service.opts.vconf = self._vconf
                    if self._pconf is not None:
                        traffic_service.opts.pconf = self._pconf
                    frame = traffic_service.process_frame(frame)

        pts, time_base = await self.next_timestamp()
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
        super().stop()


async def _cleanup_peer_connection(pc: RTCPeerConnection) -> None:
    if pc in peer_connections:
        peer_connections.remove(pc)
    for sender in pc.getSenders():
        track = getattr(sender, "track", None)
        if isinstance(track, ALPRWebRTCVideoTrack):
            track.close()
    extra_tracks = getattr(pc, "_app_tracks", [])
    for track in extra_tracks:
        if isinstance(track, ALPRWebRTCVideoTrack):
            track.close()
    if hasattr(pc, "_app_tracks"):
        pc._app_tracks = []  # type: ignore[attr-defined]
    await pc.close()


@app.get("/api/video")
def video_stream(url: str):
    return StreamingResponse(gen_frames(url, False), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/alpr_stream")
def alpr_stream(
    url: str,
    vconf: Optional[float] = None,
    pconf: Optional[float] = None,
    read_plate: Optional[bool] = None,
):
    return StreamingResponse(
        gen_frames(url, True, vconf=vconf, pconf=pconf, read_plate=read_plate),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/parking_stream")
def parking_stream(
    url: str,
    pconf: Optional[float] = None,
):
    return StreamingResponse(
        gen_parking_frames(url, pconf=pconf),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _parse_optional_float(value: Optional[object], field: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid {field} value")


def _parse_optional_bool(value: Optional[object], field: str) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise HTTPException(status_code=400, detail=f"Invalid {field} value")


@app.post("/api/webrtc/offer")
async def webrtc_offer(payload: dict):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Stream URL is required")

    mode = (payload.get("mode") or "alpr").strip().lower()
    process = mode != "preview"
    parking_mode = mode == "parking"

    offer_sdp = payload.get("sdp")
    offer_type = payload.get("type")
    if not offer_sdp or not offer_type:
        raise HTTPException(status_code=400, detail="SDP offer is required")

    if offer_type != "offer":
        raise HTTPException(status_code=400, detail="SDP type must be 'offer'")

    vconf = _parse_optional_float(payload.get("vconf"), "vconf")
    pconf = _parse_optional_float(payload.get("pconf"), "pconf")
    read_plate = _parse_optional_bool(payload.get("read_plate"), "read_plate")

    try:
        track = ALPRWebRTCVideoTrack(
            url=url,
            process=process,
            vconf=vconf,
            pconf=pconf,
            read_plate=read_plate,
            parking_mode=parking_mode,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=500, detail="Failed to initialize video track") from exc

    pc = RTCPeerConnection()
    peer_connections.add(pc)
    pc._app_tracks = [track]  # type: ignore[attr-defined]

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer_connection(pc)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        pc.addTrack(track)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as exc:
        track.close()
        await _cleanup_peer_connection(pc)
        raise HTTPException(status_code=500, detail=f"WebRTC negotiation failed: {exc}") from exc

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    is_video = (
        "video" in content_type
        or content_type == "application/octet-stream"
        or filename.endswith(".mp4")
        or filename.endswith(".mpeg")
    )
    if not is_video:
        raise HTTPException(status_code=400, detail="Only MP4 video files are accepted")

    video_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{video_id}.mp4"
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="File exceeds 200 MB limit")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save video") from exc

    with uploaded_videos_lock:
        uploaded_videos[video_id] = dest
    return {"video_id": video_id}


def _gen_frames_upload(
    path: Path,
    video_id: str,
    vconf: Optional[float] = None,
    pconf: Optional[float] = None,
    read_plate: Optional[bool] = None,
):
    try:
        yield from gen_frames(str(path), True, vconf=vconf, pconf=pconf, read_plate=read_plate)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        with uploaded_videos_lock:
            uploaded_videos.pop(video_id, None)


@app.get("/api/alpr_stream/upload/{video_id}")
def stream_uploaded_video(
    video_id: str,
    vconf: Optional[float] = None,
    pconf: Optional[float] = None,
    read_plate: Optional[bool] = None,
):
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    with uploaded_videos_lock:
        path = uploaded_videos.get(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Video not found or expired")
    return StreamingResponse(
        _gen_frames_upload(path, video_id, vconf=vconf, pconf=pconf, read_plate=read_plate),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/alpr")
@app.post("/api/alpr/")
async def alpr(
    file: UploadFile = File(...),
    mode: Optional[str] = Form(None),
    mode_query: Optional[str] = Query(None, alias="mode"),
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    requested_mode = (mode or mode_query or "").strip().lower()
    if requested_mode == "parking":
        with PARKING_PROCESS_LOCK:
            result = process_parking_frame(img)
        _, buffer = cv2.imencode('.jpg', result)
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"X-Analysis-Mode": "parking"},
        )
    with ALPR_PROCESS_LOCK:
        traffic_service.read_plate = True
        setattr(traffic_service.opts, "read_plate", True)
        result = traffic_service.process_image(img)
    _, buffer = cv2.imencode('.jpg', result)
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Analysis-Mode": "traffic"},
    )


@app.api_route("/api/parking", methods=["GET", "POST"])
@app.api_route("/api/parking/", methods=["GET", "POST"])
async def parking(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    with PARKING_PROCESS_LOCK:
        result = process_parking_frame(img)
    _, buffer = cv2.imencode('.jpg', result)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/api/routes")
def api_routes():
    routes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api"):
            continue
        methods = sorted(getattr(route, "methods", []) or [])
        routes.append({"path": path, "methods": methods})
    return {"routes": routes}


 # Serve frontend after registering API routes so it doesn't shadow them


@app.get("/api/cameras")
def camera_presets():
    try:
        with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:  # pragma: no cover - simple IO guard
        raise HTTPException(status_code=404, detail="Camera preset file missing") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - simple IO guard
        raise HTTPException(status_code=500, detail="Camera preset file invalid") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Camera preset file must be a mapping")

    presets = [{"label": key, "url": value} for key, value in data.items()]
    return {"presets": presets}


@app.get("/api/vehicle_models")
def vehicle_models():
    weights = _find_vehicle_weights()
    selected_path = Path(getattr(traffic_service.opts, "vehicle_weight", ""))
    selected_id: Optional[str] = None
    try:
        selected_id = selected_path.relative_to(VEHICLE_WEIGHTS_DIR).as_posix()
    except Exception:
        try:
            resolved = selected_path.resolve()
            selected_id = resolved.relative_to(VEHICLE_WEIGHTS_DIR.resolve()).as_posix()
        except Exception:
            selected_id = selected_path.name or None

    models = [_serialize_weight_path(path) for path in weights]
    return {"models": models, "selected": selected_id}


@app.post("/api/vehicle_models/select")
def select_vehicle_model(payload: dict):
    weight_id = payload.get("weight")
    if not weight_id or not isinstance(weight_id, str):
        raise HTTPException(status_code=400, detail="Field 'weight' is required")

    candidate_path = Path(weight_id)
    if candidate_path.is_absolute():
        # Only allow weights inside the configured weights directory
        try:
            candidate_path = candidate_path.relative_to(VEHICLE_WEIGHTS_DIR)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Weight must be inside the weights directory") from exc

    resolved_path = (VEHICLE_WEIGHTS_DIR / candidate_path).resolve()
    try:
        resolved_path.relative_to(VEHICLE_WEIGHTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid weight path") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Vehicle weight not found")
    if resolved_path.suffix.lower() not in VEHICLE_WEIGHT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported weight format")
    if "vehicle" not in resolved_path.stem.lower():
        raise HTTPException(status_code=400, detail="Not a vehicle weight")

    with ALPR_PROCESS_LOCK:
        try:
            traffic_service.set_vehicle_weight(str(resolved_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load vehicle model: {exc}") from exc

    selected_rel = resolved_path.relative_to(VEHICLE_WEIGHTS_DIR).as_posix()
    return {"selected": selected_rel}





@app.on_event("shutdown")
async def shutdown_webapp() -> None:
    remaining = list(peer_connections)
    for pc in remaining:
        await _cleanup_peer_connection(pc)

# Keep this last so it doesn't intercept /api/* routes
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
