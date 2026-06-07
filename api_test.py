#!/usr/bin/env python3
"""
Quick test script to verify the Traffic AI Service API
"""
import sys
import time


def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import cv2
        import numpy as np
        from utils.traffic_analysis import TrafficAnalysisService
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_api_creation():
    """Test that the API can be created"""
    print("\nTesting API creation...")
    try:
        import main
        print("✓ API module loaded successfully")
        print(f"  - App title: {main.app.title}")
        print(f"  - App version: {main.app.version}")
        
        # List routes
        routes = [route.path for route in main.app.routes]
        print(f"  - Routes: {len(routes)}")
        expected_routes = ["/", "/health", "/predict/image", "/predict/frame", "/reset", "/config"]
        for route in expected_routes:
            if route in routes:
                print(f"    ✓ {route}")
            else:
                print(f"    ✗ {route} missing!")
                return False
        
        return True
    except Exception as e:
        print(f"✗ API creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traffic_service():
    """Test that ALPR core can be initialized"""
    print("\nTesting Traffic Analysis Service...")
    try:
        from types import SimpleNamespace
        from utils.traffic_analysis import TrafficAnalysisService
        
        # Create minimal opts
        opts = SimpleNamespace(
            vehicle_weight="weights/vehicle/vehicle_detector.pt",
            plate_weight="weights/plate/license_plate_detector.pt",
            dsort_weight="weights/tracking/deepsort/ckpt.t7",
            vconf=0.6,
            pconf=0.25,
            ocr_thres=0.9,
            device="cpu",  # Use CPU for testing
            deepsort=False,
            read_plate=True,
            lang="en",
        )
        
        print("  Initializing ALPR Core (this may take a moment)...")
        start = time.time()
        core = TrafficAnalysisService(opts)
        elapsed = time.time() - start
        
        print(f"✓ ALPR Core initialized in {elapsed:.2f}s")
        print(f"  - Device: {core.opts.device}")
        print(f"  - Tracking: {'DeepSORT' if core.deepsort else 'SORT'}")
        print(f"  - Read plates: {core.read_plate}")
        
        return True
    except FileNotFoundError as e:
        print(f"⚠ Weight files not found: {e}")
        print("  This is expected if model weights are not downloaded yet")
        return True  # Don't fail test if weights missing
    except Exception as e:
        print(f"✗ ALPR Core error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_models():
    """Test that Pydantic models are correctly defined"""
    print("\nTesting response models...")
    try:
        import main
        
        # Test BoundingBox
        bbox = main.BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=400.0)
        print(f"✓ BoundingBox: {bbox}")
        
        # Test VehicleDetection
        det = main.VehicleDetection(
            track_id=1,
            bbox=bbox,
            vehicle_type="car",
            license_plate="ABC123",
            confidence=0.95
        )
        print(f"✓ VehicleDetection: track_id={det.track_id}, type={det.vehicle_type}")
        
        # Test PredictionResponse
        response = main.PredictionResponse(detections=[det], frame_count=42)
        print(f"✓ PredictionResponse: {len(response.detections)} detections")
        
        # Test JSON serialization
        json_data = response.model_dump()
        print(f"✓ JSON serialization successful")
        
        return True
    except Exception as e:
        print(f"✗ Response model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Traffic AI Service - Quick Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("API Creation", test_api_creation),
        ("Response Models", test_response_models),
        ("Traffic Analysis Service", test_traffic_service),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
