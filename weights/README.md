# Model Weights

This directory contains trained model weights for the TrafficCam system.

## Directory Structure

```
weights/
├── vehicle/              # Vehicle detection models
│   ├── vehicle_detector.pt
│   └── vehicle_detector.onnx
├── plate/                # License plate detection models
│   ├── license_plate_detector.pt
│   └── license_plate_detector.onnx
└── ocr/                  # OCR (text recognition) models
    └── ppocrv4/

```

## Model Categories

### Vehicle Detection Models
YOLO-based models trained to detect various vehicle types (cars, motorcycles, trucks, buses).
- **YOLO11**: Latest generation (11/2025 versions recommended)
- **YOLO12**: Newer generation (12/2025 versions recommended)
- **YOLOv8**: Stable baseline
- **YOLOv9**: Advanced features

**Recommended for production**: `vehicle_detector.pt`

### Plate Detection Models
Specialized models for detecting license plates in various conditions.
- **Resolution**: 320x320 for speed, 640x640 for accuracy
- **Format**: PyTorch (.pt) and ONNX (.onnx) for different deployment scenarios

**Recommended for production**: `license_plate_detector.pt`

### Pre-trained Models
Base YOLO models from Ultralytics, used as starting points for fine-tuning.

### Tracking Models
DeepSORT tracker weights for vehicle tracking across frames.

### OCR Models
PaddleOCR v4 models for license plate text recognition.

## Usage

Models are loaded in the main application through command-line arguments:

```bash
python main.py \
  --vehicle_weight weights/vehicle/vehicle_detector.pt \
  --plate_weight weights/plate/license_plate_detector.pt \
  --dsort_weight weights/tracking/deepsort/ckpt.t7 \
  --ocr_weight weights/ocr/ppocrv4/
```

## Model Naming Convention

Format: `{task}_{architecture}_{resolution}_{date}.{ext}`

- **task**: vehicle, plate
- **architecture**: yolo11n/s/m, yolo12n/s/m, yolov8n/s/m, yolov9s/c/e
  - n: nano (fastest)
  - s: small (balanced)
  - m: medium (accurate)
  - c: compact
  - e: extra-large
- **resolution**: 320, 640 (input image size)
- **date**: training/release date (DDMMMYYYY or YYYYMMDD)
- **ext**: pt (PyTorch), onnx (ONNX Runtime)

## Adding New Models

1. Train your model following the guidelines in the main README
2. Save the model with the proper naming convention
3. Place it in the appropriate subdirectory
4. Update this README with model information
5. Test with the application

## Model Performance

For detailed performance metrics and benchmarks, see [MODEL_ZOO.md](../MODEL_ZOO.md) in the project root.

## Notes

- ONNX models are optimized for deployment and faster inference
- PyTorch models (.pt) are required for training and fine-tuning
- Keep both formats for flexibility in deployment scenarios
- OCR models are maintained separately by PaddleOCR
