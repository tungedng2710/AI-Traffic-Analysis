import json
from contextlib import asynccontextmanager
from pathlib import Path

from anyio import to_thread
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from vlm.ollama_model import get_vlm, prepare_image_bytes


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = PROJECT_ROOT / "data" / "assets"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples" / "test_samples"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _warmup_model()
    yield


app = FastAPI(
    title="License Plate Reader",
    description="Image-based license plate reading service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


class HealthResponse(BaseModel):
    status: str
    ready: bool


class SampleImage(BaseModel):
    name: str
    url: str


class PredictionResponse(BaseModel):
    text: str
    elapsed_ms: int
    prediction: dict[str, str]
    vehicle_type: str
    vehicle_color: str
    license_plate: str


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    vlm = get_vlm()
    try:
        available_models = vlm.list_models()
    except Exception:
        available_models = []
    model_available = vlm.model_id in available_models
    return HealthResponse(
        status="ready" if model_available else "unavailable",
        ready=model_available,
    )


@app.get("/api/samples", response_model=list[SampleImage])
async def samples():
    return [
        SampleImage(name=path.name, url=f"/api/samples/{path.name}")
        for path in _sample_image_paths()
    ]


@app.get("/api/samples/{sample_path:path}", response_class=FileResponse)
async def sample_image(sample_path: str):
    path = (SAMPLES_DIR / sample_path).resolve()
    if not _is_safe_sample_path(path):
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(path)


@app.post("/api/read-plate", response_model=PredictionResponse)
async def read_plate(
    file: UploadFile = File(...),
):
    image_bytes = await _read_upload_image(file)

    try:
        result = await to_thread.run_sync(get_vlm().generate, image_bytes)
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Missing Python dependency: {exc.name}.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    prediction = _parse_prediction(result.text)
    return PredictionResponse(
        text=result.text,
        elapsed_ms=result.elapsed_ms,
        prediction=prediction,
        vehicle_type=prediction["vehicle_type"],
        vehicle_color=prediction["vehicle_color"],
        license_plate=prediction["license_plate"],
    )


async def _warmup_model() -> None:
    try:
        await to_thread.run_sync(get_vlm().warmup)
    except Exception:
        pass


def _sample_image_paths() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return [
        path
        for path in sorted(SAMPLES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


async def _read_upload_image(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        return prepare_image_bytes(await file.read())
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def _is_safe_sample_path(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and SAMPLES_DIR.resolve() in path.parents
    )


def _parse_prediction(text: str) -> dict[str, str]:
    defaults = {
        "vehicle_type": "unknown",
        "vehicle_color": "unknown",
        "license_plate": "unknown",
    }

    try:
        data = json.loads(_extract_json_text(text))
    except json.JSONDecodeError:
        return defaults

    if not isinstance(data, dict):
        return defaults

    prediction = {
        str(key).strip(): _clean_prediction_value(value, "unknown")
        for key, value in data.items()
        if str(key).strip()
    }

    return defaults | prediction


def _clean_prediction_value(value: object, default: str) -> str:
    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    value = str(value).strip()
    return value or default


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
