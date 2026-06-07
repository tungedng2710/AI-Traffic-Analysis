<p align="center">
  <img src="https://raw.githubusercontent.com/tungedng2710/tungedng2710.github.io/main/assets/images/tonai_logo.png" alt="TonAI Vision Lab" width="200" style="border-radius: 80px;">
</p>

<h1 align="center">TonAI Vision Lab - TrafficCam</h1>

<p align="center">
  <strong>License Plate Recognition (ALPR) and Traffic Surveillance System</strong>
</p>

![Demo](data/assets/demo.jpg)
![Demo](data/assets/demo2.png)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Service](#api-service)
  - [Configuration](#configuration)
  - [API Endpoints](#api-endpoints)
  - [Usage Examples](#usage-examples)
- [Training Models](#training-models)
- [Model Zoo](#model-zoo)
- [Model Weights](#model-weights)
- [Data Management](#data-management)
- [Dataset Download](#dataset-download)
- [WebApp Interface](#webapp-interface)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Citation](#citation)

---

## Overview

TrafficCam is a comprehensive traffic surveillance and license plate recognition system built on modern deep learning technologies. The system has been refactored into a clean, modular architecture:

- **FastAPI Service**: JSON-based REST API for vehicle and license plate detection
- **Core ALPR Engine**: Shared processing logic for detection, tracking, and OCR
- **Web Application**: Real-time visualization interface (beta)
- **Modular Design**: Clean separation between API, processing, and visualization

**Latest Updates:**
- **January 2026**: Refactored to FastAPI service with JSON API
- **August 2025**: PPOCRv5 and Web App (beta version) for testing
- **July 2024**: PPOCRv4 with text detection for improved plate recognition
- **November 2023**: TensorRT inference and ONNX model support

---

## Features

### Completed ✅
- **Vehicle Detection**: YOLO-based multi-class vehicle detection (cars, trucks, buses, motorcycles, bicycles)
- **License Plate Detection**: High-accuracy plate detection in various conditions
- **OCR Recognition**: PaddleOCR v4/v5 for text recognition
- **Object Tracking**: SORT and DeepSORT tracking algorithms
- **Multiple Formats**: PyTorch (.pt) and ONNX (.onnx) model support
- **TensorRT Optimization**: Accelerated inference with TensorRT
- **REST API**: FastAPI-based service with JSON responses
- **Web Interface**: Real-time visualization with WebRTC support

### In Progress 🚧
- Vision Language Model integration
- AutoLabel Pipeline

---

## Prerequisites

- **Operating System**: Ubuntu 20.04 or later (Linux recommended)
- **Python**: 3.9 - 3.12
- **GPU**: NVIDIA GPU with CUDA support (recommended for production)
- **Storage**: Sufficient space for model weights (~2GB)

---

## Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Start both services with one command
docker compose up -d

# Or use the helper script
./docker.sh start
```

**Services will be available at:**
- Analysis API: http://localhost:7862 (Swagger: http://localhost:7862/docs)
- WebApp: http://localhost:7863

### Option 2: Manual Installation

### 1. Installation

```bash
# Install dependencies
bash scripts/install.sh

# Or manually
pip install -r requirements.txt
```

### 2. Start API Service

```bash
# Quick start
./quickstart.sh

# Or manually
python main.py

# Using Uvicorn (production)
uvicorn main:app --host 0.0.0.0 --port 7862
```

The service will start on `http://localhost:7862`

### 3. Test the Service

```bash
# Health check
curl http://localhost:7862/health

# Process an image
curl -X POST http://localhost:7862/predict/image \
  -F "file=@path/to/image.jpg"

# Using example client
python client_example.py --health
python client_example.py --image path/to/image.jpg
python client_example.py --video path/to/video.mp4
```

### 4. Interactive API Documentation
- **Swagger UI**: http://localhost:7862/docs
- **ReDoc**: http://localhost:7862/redoc

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Traffic AI System                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                 ┌─────────────────┐    │
│  │   main.py    │◄────────────────┤ client_example  │    │
│  │ FastAPI API  │                 │    (Python)     │    │
│  │   Service    │                 └─────────────────┘    │
│  │              │                                        │
│  │ Endpoints:   │                                        │
│  │ /predict/    │                                        │
│  │   image      │                                        │
│  │ /predict/    │                                        │
│  │   frame      │                                        │
│  │ /reset       │                                        │
│  │ /config      │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         │ uses                                           │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │ utils/         │                                      │
│  │ traffic_       │   Core Processing Logic              │
│  │  analysis.py   │   - Vehicle Detection (YOLO)         │
│  │                │   - Plate Detection (YOLO)           │
│  │                │   - OCR (PaddleOCR)                  │
│  │                │   - Tracking (SORT/DeepSORT)         │
│  └────────────────┘                                      │
│                                                          │
│  ┌─────────────────────────────────────────┐             │
│  │        webapp/ (Visualization)          │             │
│  │  ┌─────────────────┐  ┌──────────────┐  │             │
│  │  │ backend/main.py │  │  frontend/   │  │             │
│  │  │ - Video Stream  │  │  - Web UI    │  │             │
│  │  │ - WebRTC        │  │  - Player    │  │             │
│  │  │ - RTSP Support  │  │              │  │             │
│  │  └─────────────────┘  └──────────────┘  │             │
│  └─────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
license-plate-recognition/
├── main.py                    # FastAPI service (primary entry point)
├── client_example.py          # Example client for testing
├── api_test.py                # API testing script
├── requirements.txt           # Python dependencies
├── data/                      # Data management
│   ├── assets/               # Demo images and visual resources
│   ├── config/               # Configuration files (JSON, TXT)
│   ├── logs/                 # Application logs and outputs
│   ├── recordings/           # Camera recording data
│   ├── results/              # Processing results and reports
│   ├── samples/              # Test and raw sample data
│   └── scripts/              # Data processing utilities
├── weights/                   # Model weights
│   ├── vehicle/              # Vehicle detection models
│   ├── plate/                # License plate detection models
│   ├── pretrained/           # Pre-trained YOLO base models
│   ├── tracking/             # DeepSORT tracker weights
│   └── ocr/                  # PaddleOCR models
├── utils/                     # Core utilities
│   ├── traffic_analysis.py   # Traffic Analysis Service (core logic)
│   ├── ocr.py                # OCR utilities
│   ├── utils.py              # Helper functions
│   └── ppocr_configs.yaml    # PaddleOCR configuration
├── tracking/                  # Object tracking algorithms
│   ├── sort.py               # SORT tracker
│   └── deep_sort/            # DeepSORT tracker
├── detectors/                 # Detector implementations
│   ├── yolo/                 # Ultralytics YOLO (v8, v9, v10, v11)
│   └── rf-detr/              # RF-DETR transformer detector
├── scripts/                   # Training and testing scripts
└── webapp/                    # Web visualization application (beta)
    ├── backend/              # FastAPI backend for streaming
    ├── frontend/             # Web UI
    └── utils/                # Webapp utilities
```

---

## API Service

### Configuration

The service is configured using environment variables:

```bash
# Device configuration
export ALPR_DEVICE=cuda:0      # Options: auto, cpu, cuda:0, cuda:1, etc.

# Model paths
export VEHICLE_WEIGHT=weights/vehicle/vehicle_detector.pt
export PLATE_WEIGHT=weights/plate/license_plate_detector.pt
export DSORT_WEIGHT=weights/tracking/deepsort/ckpt.t7

# Detection thresholds
export VEHICLE_CONF=0.6        # Vehicle detection confidence (0.0-1.0)
export PLATE_CONF=0.25         # Plate detection confidence (0.0-1.0)
export OCR_THRESHOLD=0.9       # OCR confidence threshold (0.0-1.0)

# Tracking
export USE_DEEPSORT=false      # Use DeepSORT (true) or SORT (false)

# Language
export LANG=en                 # Options: en, vi, es, fr

# Server
export HOST=0.0.0.0
export PORT=7862
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information and available endpoints |
| `/health` | GET | Health check and configuration status |
| `/predict/image` | POST | Process single image (resets tracking) |
| `/predict/frame` | POST | Process video frame (maintains tracking) |
| `/reset` | POST | Reset tracker state |
| `/config` | GET | Get current configuration |

### Response Format

```json
{
  "detections": [
    {
      "track_id": 1,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 350.8,
        "y2": 450.2
      },
      "vehicle_type": "car",
      "license_plate": "ABC123",
      "plate_bbox": {
        "x1": 150.0,
        "y1": 380.0,
        "x2": 300.0,
        "y2": 420.0
      },
      "confidence": 0.95
    }
  ],
  "frame_count": 42
}
```

### Usage Examples

#### Using cURL

```bash
# Health check
curl http://localhost:7862/health

# Process image
curl -X POST http://localhost:7862/predict/image \
  -F "file=@car.jpg"

# Process video frame
curl -X POST "http://localhost:7862/predict/frame?frame_number=42" \
  -F "file=@frame.jpg"

# Reset tracker
curl -X POST http://localhost:7862/reset

# Get configuration
curl http://localhost:7862/config
```

#### Using Python

```python
import requests

# Process single image
with open("car.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:7862/predict/image",
        files={"file": f}
    )
    data = response.json()
    for det in data['detections']:
        print(f"Vehicle: {det['vehicle_type']}, Plate: {det.get('license_plate', 'N/A')}")

# Process video
import cv2

cap = cv2.VideoCapture("video.mp4")
frame_num = 0

# Reset tracker before processing
requests.post("http://localhost:7862/reset")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Send to API
    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    params = {"frame_number": frame_num}
    response = requests.post(
        "http://localhost:7862/predict/frame",
        files=files,
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Frame {frame_num}: {len(data['detections'])} vehicles detected")
    
    frame_num += 1

cap.release()
```

#### Using Example Client

```bash
# Health check
python client_example.py --health

# Process image
python client_example.py --image path/to/image.jpg

# Process video (all frames)
python client_example.py --video path/to/video.mp4

# Process limited frames
python client_example.py --video path/to/video.mp4 --max-frames 100

# Get configuration
python client_example.py --config
```

---

## Training Models

### Train Ultralytics YOLO

Train a detector directly from the CLI:

```bash
# Using bash wrapper
./scripts/train_yolo.sh \
  --data data/Peru_License_Plate/data.yaml \
  --model yolov8n.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 320 \
  --device 0

# Or using Python directly
python detectors/yolo/train_ultralytics.py \
  --data data/Peru_License_Plate/data.yaml \
  --model yolov8n.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 320 \
  --device 0
```

**Common Arguments:**
- `--data`: Path to `data.yaml` (e.g., `data/LP-11k/data.yaml`)
- `--model`: YOLO config or weights (e.g., `yolov8n.yaml`, `yolov8s.pt`)
- `--epochs`, `--batch`, `--imgsz`: Training hyperparameters
- `--device`: CUDA device id(s) or `cpu`
- `--project`, `--name`: Output directory configuration
- `--exist-ok`, `--workers`, `--resume`, `--patience`, `--seed`: Additional options

### Export to ONNX

Convert trained models to ONNX format for deployment:

```bash
python detectors/yolo/exporter.py \
  --weights weights/plate/license_plate_detector.pt \
  --dynamic
```

**Arguments:**
- `--weights`: Path to the `.pt` file (relative or absolute)
- `--dynamic`: Keep dynamic batch/sequence axes (optional)
- `--imgsz`, `--half`: Additional export options

Place exported `.onnx` files in the `weights/` directory.

---

## Model Zoo

Current best-performing models trained on various datasets:

| Model | mAP50 | mAP50-95 | Precision | Recall | Dataset |
|-------|-------|----------|-----------|--------|---------|
| YOLO11m | 0.875 | 0.596 | 0.941 | 0.802 | vehicle_18sep2025 |
| YOLO11n | 0.847 | 0.493 | 0.844 | 0.765 | vehicle_18sep2025 |
| YOLO12m | 0.845 | 0.580 | 0.817 | 0.777 | vehicle_18sep2025 |
| YOLO12s | 0.818 | 0.490 | 0.800 | 0.763 | vehicle_18sep2025 |
| YOLOv9e | 0.770 | 0.535 | 0.837 | 0.718 | vehicle_30oct2025 |

Generate updated model zoo report:

```bash
python scripts/generate_model_zoo.py --runs-dir runs/detect --output MODEL_ZOO.md
```

---

## Model Weights

### Directory Structure

```
weights/
├── vehicle/              # Vehicle detection models
│   ├── vehicle_detector.pt
│   └── vehicle_detector.onnx
├── plate/                # License plate detection models
│   ├── license_plate_detector.pt
│   └── license_plate_detector.onnx
└── ocr/                  # OCR models (PaddleOCR)
    └── ppocrv4/
```

### Model Naming Convention

Format: `{task}_{architecture}_{resolution}_{date}.{ext}`

- **task**: vehicle, plate
- **architecture**: yolo11n/s/m, yolo12n/s/m, yolov8n/s/m, yolov9s/c/e
  - n: nano (fastest)
  - s: small (balanced)
  - m: medium (accurate)
  - c: compact
  - e: extra-large
- **resolution**: 320, 640 (input image size)
- **date**: Training/release date
- **ext**: pt (PyTorch), onnx (ONNX Runtime)

### Recommended Models

**Production Use:**
- Vehicle: `vehicle_detector.pt`
- Plate: `license_plate_detector.pt`

---

## Data Management

### Directory Structure

```
data/
├── assets/              # Demo images and visual assets
├── config/              # Configuration files (JSON, TXT)
│   ├── rtsp_url.json   # RTSP camera URLs
│   └── streams.txt     # Stream configuration
├── logs/                # Application logs and outputs
│   ├── frames/         # Extracted video frames
│   ├── objects/        # Detected objects
│   └── plates/         # License plate detections
├── recordings/          # Camera recording data
│   ├── cam_thai_nguyen/
│   └── cyhome_bike/
├── results/             # Processing results and reports
├── samples/             # Sample data for testing
│   ├── raw/            # Raw unprocessed samples
│   └── test_samples/   # Test dataset samples
└── scripts/             # Data processing utilities
    ├── crop.py
    ├── download_roboflow.py
    ├── record_rtsp_streams.py
    └── yt_downloader.py
```

### Usage Guidelines

- Place camera recordings in `recordings/<camera_name>/`
- Store configuration files in `config/`
- Processing results are saved to `results/`
- Logs are automatically generated in `logs/`
- Use scripts in `scripts/` for data operations

---

## Dataset Download

### Roboflow Integration

Automate dataset downloads from Roboflow:

1. **Install Roboflow**:
```bash
pip install roboflow
```

2. **Configure `.env`**:
```bash
ROBOFLOW_API_KEY=your-api-key
ROBOFLOW_WORKSPACE=your-workspace
ROBOFLOW_PROJECT=your-project
ROBOFLOW_VERSION=1
ROBOFLOW_EXPORT_FORMAT=yolov8          # optional, defaults to yolov8
ROBOFLOW_POLL_INTERVAL=600             # optional, seconds between checks
```

3. **Run downloader**:
```bash
python data/scripts/download_roboflow.py
```

The script checks Roboflow every 10 minutes (configurable), downloads new versions, and auto-increments the version number.

---

## WebApp Interface

### Start WebApp

```bash
cd webapp
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7863
```

Open `http://localhost:7863` in your browser.

### Features

- Real-time video stream visualization
- RTSP camera support
- WebRTC streaming
- Camera presets management
- Model selection interface
- Drawing overlays (bounding boxes, labels, plates)

**Note**: The webapp is independent from the main API service and uses its own processing instance.

---

## Docker Deployment

### Quick Start with Docker Compose

Run both services (Analysis API + WebApp) with a single command:

```bash
# Production mode (detached)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Using Helper Script:**

```bash
./docker.sh start       # Start services
./docker.sh logs        # View logs
./docker.sh status      # Check status
./docker.sh stop        # Stop services
./docker.sh help        # Show all commands
```

**Services:**
- Analysis Service API: http://localhost:7862
  - Swagger UI: http://localhost:7862/docs
  - ReDoc: http://localhost:7862/redoc
- WebApp Interface: http://localhost:7863

### Development Mode

For hot-reloading during development:

```bash
# Start in development mode
docker compose -f docker-compose.dev.yml up

# Or detached
docker compose -f docker-compose.dev.yml up -d
```

### Manual Docker Build

```bash
# Build image
docker build -t trafficcam .

# Run analysis service
docker run -d \
  -p 7862:7862 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  --name traffic-analysis \
  trafficcam

# Run webapp
docker run -d \
  -p 7863:7863 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  --name traffic-webapp \
  trafficcam \
  uvicorn webapp.backend.main:app --host 0.0.0.0 --port 7863
```

### GPU Support

The docker-compose files include GPU support. If you don't have GPU, remove the `deploy` section from docker-compose.yml or set `ALPR_DEVICE=cpu`.

**Requirements:**
- NVIDIA Docker runtime installed
- NVIDIA GPU with CUDA support

---

## Development

### Testing

```bash
# Test imports and API creation
python api_test.py

# Test with example client
python client_example.py --health
python client_example.py --config

# Test with sample data
python client_example.py --image data/samples/test.jpg
python client_example.py --video data/recordings/test.mp4 --max-frames 100

# Quick ALPR test
python scripts/test_alpr.py --input_source path/to/image_or_video
```

### Adding New Models

1. Train your model following the training guidelines
2. Save with proper naming convention
3. Place in appropriate `weights/` subdirectory
4. Update configuration if needed
5. Test with the API

---

## Citation

If this project helps your research or deployment, please cite:

```bibtex
@misc{trafficcam2025,
  title        = {TrafficCam: Traffic Surveillance and License Plate Recognition},
  author       = {Nguyen, Tung},
  year         = {2025},
  howpublished = {\url{https://github.com/tungedng2710/license-plate-recognition}}
}
```

---

## License

Copyright (C) 2023-2026 TonAI Vision Lab

---

## Support

For questions and issues:
- Check the [Interactive API Documentation](http://localhost:7862/docs) when service is running
- Review example code in [client_example.py](client_example.py)
- Check existing issues or contact the development team

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection models
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for text recognition
- [DeepSORT](https://github.com/John1liu/YOLOV5-DeepSORT-Vehicle-Tracking-Master) for tracking implementation
