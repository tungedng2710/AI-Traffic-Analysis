"""
Copyright (C) 2023 TonAI
Traffic AI Service - FastAPI Backend for License Plate Recognition
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from types import SimpleNamespace
import os

from utils.traffic_analysis import TrafficAnalysisService

# Initialize FastAPI
app = FastAPI(
    title="Traffic AI Service",
    description="License Plate Recognition API for vehicle detection and tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default device configuration
DEFAULT_DEVICE = os.environ.get("ALPR_DEVICE", "auto")

# Initialize ALPR Core
opts = SimpleNamespace(
    vehicle_weight=os.environ.get("VEHICLE_WEIGHT", "weights/vehicle/vehicle_detector.pt"),
    plate_weight=os.environ.get("PLATE_WEIGHT", "weights/plate/license_plate_detector.pt"),
    dsort_weight=os.environ.get("DSORT_WEIGHT", "weights/tracking/deepsort/ckpt.t7"),
    vconf=float(os.environ.get("VEHICLE_CONF", "0.6")),
    pconf=float(os.environ.get("PLATE_CONF", "0.25")),
    ocr_thres=float(os.environ.get("OCR_THRESHOLD", "0.9")),
    device=DEFAULT_DEVICE,
    deepsort=os.environ.get("USE_DEEPSORT", "false").lower() == "true",
    read_plate=True,
    lang=os.environ.get("LANG", "en"),
)

traffic_service = TrafficAnalysisService(opts)


# Pydantic models for API responses
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class VehicleDetection(BaseModel):
    track_id: int
    bbox: BoundingBox
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None
    plate_bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    detections: List[VehicleDetection]
    frame_count: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Traffic AI - License Plate Recognition",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict/image": "Process single image (POST)",
            "/predict/frame": "Process video frame (POST)",
            "/reset": "Reset tracker (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=str(traffic_service.opts.device),
        models_loaded=True
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Process a single image and return vehicle detections with license plates
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        JSON with list of detections containing track_id, bbox, vehicle_type, license_plate
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Reset tracker for single image processing
        traffic_service.reset()
        
        # Process frame (without drawing)
        _ = traffic_service.process_frame(img)
        
        # Extract detections
        detections = []
        for track_id, vehicle in traffic_service.vehicles.items():
            bbox_xyxy = vehicle.bbox_xyxy
            
            detection = VehicleDetection(
                track_id=track_id,
                bbox=BoundingBox(
                    x1=float(bbox_xyxy[0]),
                    y1=float(bbox_xyxy[1]),
                    x2=float(bbox_xyxy[2]),
                    y2=float(bbox_xyxy[3])
                ),
                vehicle_type=vehicle.vehicle_type if vehicle.vehicle_type else None,
                license_plate=vehicle.plate_number if vehicle.plate_number else None,
                confidence=float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None
            )
            
            # Add plate bbox if available
            if vehicle.license_plate_bbox is not None:
                plate_bbox = vehicle.license_plate_bbox
                detection.plate_bbox = BoundingBox(
                    x1=float(plate_bbox[0]),
                    y1=float(plate_bbox[1]),
                    x2=float(plate_bbox[2]),
                    y2=float(plate_bbox[3])
                )
            
            detections.append(detection)
        
        return PredictionResponse(detections=detections)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/predict/frame", response_model=PredictionResponse)
async def predict_frame(file: UploadFile = File(...), frame_number: Optional[int] = None):
    """
    Process a video frame and return vehicle detections with license plates.
    Maintains tracking state across frames.
    
    Args:
        file: Image file representing a video frame
        frame_number: Optional frame number for reference
        
    Returns:
        JSON with list of detections containing track_id, bbox, vehicle_type, license_plate
    """
    try:
        # Read frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        
        # Process frame (maintains tracking state)
        _ = traffic_service.process_frame(frame)
        
        # Extract detections
        detections = []
        for track_id, vehicle in traffic_service.vehicles.items():
            bbox_xyxy = vehicle.bbox_xyxy
            
            detection = VehicleDetection(
                track_id=track_id,
                bbox=BoundingBox(
                    x1=float(bbox_xyxy[0]),
                    y1=float(bbox_xyxy[1]),
                    x2=float(bbox_xyxy[2]),
                    y2=float(bbox_xyxy[3])
                ),
                vehicle_type=vehicle.vehicle_type if vehicle.vehicle_type else None,
                license_plate=vehicle.plate_number if vehicle.plate_number else None,
                confidence=float(vehicle.ocr_conf) if vehicle.ocr_conf > 0 else None
            )
            
            # Add plate bbox if available
            if vehicle.license_plate_bbox is not None:
                plate_bbox = vehicle.license_plate_bbox
                detection.plate_bbox = BoundingBox(
                    x1=float(plate_bbox[0]),
                    y1=float(plate_bbox[1]),
                    x2=float(plate_bbox[2]),
                    y2=float(plate_bbox[3])
                )
            
            detections.append(detection)
        
        return PredictionResponse(
            detections=detections,
            frame_count=frame_number
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/reset")
async def reset_tracker():
    """
    Reset the tracking state. Useful when starting a new video or stream.
    """
    try:
        traffic_service.reset()
        return {"status": "success", "message": "Tracker reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "vehicle_weight": traffic_service.opts.vehicle_weight,
        "plate_weight": traffic_service.opts.plate_weight,
        "device": str(traffic_service.opts.device),
        "vehicle_confidence": traffic_service.opts.vconf,
        "plate_confidence": traffic_service.opts.pconf,
        "ocr_threshold": traffic_service.ocr_thres,
        "use_deepsort": traffic_service.deepsort,
        "read_plate": traffic_service.read_plate,
        "language": traffic_service.lang
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7862")),
        log_level="info"
    )
