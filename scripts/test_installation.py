#!/usr/bin/env python
"""
Quick test script to verify the installation and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ Torchvision import failed: {e}")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    return True


def test_model():
    """Test model initialization."""
    print("\nTesting model initialization...")
    
    try:
        from model.face_attribute_model import FaceAttributeModel
        
        model = FaceAttributeModel()
        print(f"✓ Model initialized successfully")
        
        attributes = model.get_attribute_list()
        print(f"✓ Model supports {len(attributes)} attributes")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_api():
    """Test that API can be imported."""
    print("\nTesting API...")
    
    try:
        # Just test import, don't start the server
        import uvicorn
        from api.app import app
        print(f"✓ API application loaded successfully")
        return True
    except Exception as e:
        print(f"✗ API import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Face Attribute Classification - Installation Test")
    print("=" * 50)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test model
    results.append(("Model", test_model()))
    
    # Test API
    results.append(("API", test_api()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! The application is ready to use.")
        print("\nTo start the server, run:")
        print("  uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000")
        print("\nOr use Docker:")
        print("  docker-compose up --build")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check your installation.")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
