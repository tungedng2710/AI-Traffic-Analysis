import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_MODEL_ID = os.environ.get("VLM_MODEL_ID", "qwen3.5:0.8b")
DEFAULT_PROMPT = os.environ.get(
    "VLM_PROMPT",
    (
        "Analyze the vehicle image. Return only valid compact JSON with these keys: "
        'vehicle_type, vehicle_color, license_plate. Use "unknown" when a value is '
        "not visible. Do not include markdown, explanation, or extra text."
    ),
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "80"))
DEFAULT_OLLAMA_HOST = os.environ.get("VLM_OLLAMA_HOST", "http://0.0.0.0:11434")
DEFAULT_KEEP_ALIVE = os.environ.get("VLM_KEEP_ALIVE", "30m")
DEFAULT_IMAGE_MAX_SIDE = int(os.environ.get("VLM_IMAGE_MAX_SIDE", "1280"))
DEFAULT_IMAGE_QUALITY = int(os.environ.get("VLM_IMAGE_QUALITY", "85"))
THINKING_ENABLED = False
DEFAULT_SAMPLE_IMAGE = Path(
    os.environ.get(
        "VLM_SAMPLE_IMAGE",
        "/root/tungn197/license-plate-recognition/data/samples/test_samples/test_bien_so.jpg",
    )
)


@dataclass(frozen=True)
class VLMConfig:
    model_id: str = DEFAULT_MODEL_ID
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ollama_host: str = DEFAULT_OLLAMA_HOST
    keep_alive: str = DEFAULT_KEEP_ALIVE
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    image_quality: int = DEFAULT_IMAGE_QUALITY
    think: bool = THINKING_ENABLED

    def normalized(self) -> "VLMConfig":
        return VLMConfig(
            model_id=self.model_id,
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
            ollama_host=normalize_ollama_host(self.ollama_host),
            keep_alive=self.keep_alive,
            image_max_side=self.image_max_side,
            image_quality=self.image_quality,
            think=self.think,
        )


@dataclass(frozen=True)
class VLMResult:
    text: str
    raw_text: str
    elapsed_ms: int
    model_id: str
    prompt: str
    max_new_tokens: int
    ollama_host: str


class OllamaVLM:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        config: Optional[VLMConfig] = None,
    ):
        self.config = (
            config or VLMConfig(model_id=model_id, ollama_host=ollama_host)
        ).normalized()
        self.model_id = self.config.model_id
        self.ollama_host = self.config.ollama_host

    def is_available(self) -> bool:
        return self.model_id in self.list_models()

    def list_models(self) -> list[str]:
        data = self._request_json("GET", "/api/tags")
        return [model["name"] for model in data.get("models", []) if model.get("name")]

    def warmup(self) -> None:
        payload = self._build_payload(prompt="OK", max_new_tokens=1)
        self._request_json("POST", "/api/generate", payload=payload, timeout=60)

    def generate(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> VLMResult:
        prompt = prompt or self.config.prompt
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        start = time.perf_counter()
        payload = self._build_payload(prompt, max_new_tokens, image_bytes)
        data = self._request_json("POST", "/api/generate", payload=payload, timeout=180)
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        return VLMResult(
            text=(data.get("response") or "").strip(),
            raw_text=json.dumps(data, ensure_ascii=False),
            elapsed_ms=elapsed_ms,
            model_id=self.model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            ollama_host=self.ollama_host,
        )

    def _build_payload(
        self,
        prompt: str,
        max_new_tokens: int,
        image_bytes: Optional[bytes] = None,
    ) -> dict:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "think": self.config.think,
            "keep_alive": self.config.keep_alive,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0,
            },
        }
        if image_bytes:
            payload["images"] = [base64.b64encode(image_bytes).decode("ascii")]
        return payload

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout: int = 10,
    ) -> dict:
        request = Request(
            f"{self.ollama_host}{path}",
            data=_json_body(payload),
            headers=_json_headers(payload),
            method=method,
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_host}. Start Ollama with `ollama serve`."
            ) from exc


def normalize_ollama_host(host: str) -> str:
    host = host or "http://127.0.0.1:11434"
    if "://" not in host:
        host = f"http://{host}"

    parsed = urlparse(host)
    hostname = "127.0.0.1" if parsed.hostname == "0.0.0.0" else parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname or '127.0.0.1'}{port}"
    return urlunparse((parsed.scheme, netloc, "", "", "", "")).rstrip("/")


def prepare_image_bytes(
    image_bytes: bytes,
    max_side: int = DEFAULT_IMAGE_MAX_SIDE,
    quality: int = DEFAULT_IMAGE_QUALITY,
) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()


_vlm: Optional[OllamaVLM] = None


def get_vlm() -> OllamaVLM:
    global _vlm
    if _vlm is None:
        _vlm = OllamaVLM()
    return _vlm


def read_plate(
    image_path: str | Path = DEFAULT_SAMPLE_IMAGE,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> VLMResult:
    image_bytes = prepare_image_bytes(Path(image_path).read_bytes())
    return get_vlm().generate(image_bytes, prompt=prompt, max_new_tokens=max_new_tokens)


def _json_body(payload: Optional[dict]) -> Optional[bytes]:
    if payload is None:
        return None
    return json.dumps(payload).encode("utf-8")


def _json_headers(payload: Optional[dict]) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    return headers


if __name__ == "__main__":
    result = read_plate()
    print(result.text)
