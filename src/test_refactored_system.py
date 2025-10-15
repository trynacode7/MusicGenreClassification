"""
Test script for the refactored music genre classification system.

This script tests the basic functionality without requiring the full dataset.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier, GenreDashboard
        print("[OK] Main classifier classes imported successfully")
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    
    try:
        import numpy as np
        import pandas as pd
        import librosa
        import matplotlib.pyplot as plt
        import tkinter as tk
        print("[OK] All required dependencies available")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if models can be loaded."""
    print("\nTesting model loading...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("[ERROR] Models directory not found")
        return False
    
    required_files = ["scaler.pkl", "label_encoder.pkl", "random_forest_model.pkl", "svm_model.pkl"]
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        print(f"[ERROR] Missing model files: {missing_files}")
        return False
    
    print("[OK] All required model files found")
    return True

def test_classifier_initialization():
    """Test if classifier can be initialized."""
    print("\nTesting classifier initialization...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier
        classifier = MusicGenreClassifier()
        print("[OK] Classifier initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Classifier initialization failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction with synthetic audio."""
    print("\nTesting feature extraction...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier
        classifier = MusicGenreClassifier()
        
        # Create synthetic audio data
        duration = 3.0  # seconds
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Extract features
        features = classifier.extract_features(audio_data, sr)
        
        if len(features) > 0:
            print(f"[OK] Feature extraction successful (extracted {len(features)} features)")
            return True
        else:
            print("[ERROR] Feature extraction returned empty features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return False

def test_prediction():
    """Test genre prediction with synthetic audio."""
    print("\nTesting genre prediction...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier
        classifier = MusicGenreClassifier()
        
        # Create synthetic audio data
        duration = 3.0  # seconds
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test prediction with SVM (most reliable)
        result = classifier.predict_genre(audio_data, 'svm')
        
        if 'genre' in result and 'confidence' in result:
            print(f"[OK] Prediction successful: {result['genre']} (confidence: {result['confidence']:.3f})")
            return True
        else:
            print("[ERROR] Prediction failed - missing required fields")
            return False
            
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return False

def test_chunk_creation():
    """Test audio chunking functionality."""
    print("\nTesting audio chunking...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier
        classifier = MusicGenreClassifier()
        
        # Create longer synthetic audio data
        duration = 10.0  # seconds
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create chunks
        chunks = classifier.create_audio_chunks(audio_data, sr)
        
        if len(chunks) > 0:
            print(f"[OK] Chunking successful (created {len(chunks)} chunks)")
            print(f"  First chunk: {chunks[0][1]:.1f}s - {chunks[0][2]:.1f}s")
            return True
        else:
            print("[ERROR] Chunking failed - no chunks created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Chunking failed: {e}")
        return False

def test_dashboard_creation():
    """Test dashboard creation (without running)."""
    print("\nTesting dashboard creation...")
    
    try:
        from refactored_genre_classifier import MusicGenreClassifier
        classifier = MusicGenreClassifier()
        
        # Create dashboard (don't run it)
        dashboard = classifier.create_dashboard()
        
        if dashboard is not None:
            print("[OK] Dashboard created successfully")
            return True
        else:
            print("[ERROR] Dashboard creation failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Dashboard creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Refactored Music Genre Classification System - Test Suite ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Classifier Initialization Test", test_classifier_initialization),
        ("Feature Extraction Test", test_feature_extraction),
        ("Prediction Test", test_prediction),
        ("Chunking Test", test_chunk_creation),
        ("Dashboard Creation Test", test_dashboard_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] All tests passed! The refactored system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python src/example_usage.py")
        print("2. Or run: python src/refactored_genre_classifier.py")
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Ensure models are trained: python src/model_training.py")
        print("3. Check that the models directory exists and contains required files")

if __name__ == "__main__":
    main()
