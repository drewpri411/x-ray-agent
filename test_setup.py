#!/usr/bin/env python3
"""
Test script to verify the AI Radiology Assistant setup.
Run this script to check if all dependencies and components are working correctly.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchxrayvision
        print("✅ TorchXRayVision imported successfully")
    except ImportError as e:
        print(f"❌ TorchXRayVision import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain imported successfully")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import langgraph
        print("✅ LangGraph imported successfully")
    except ImportError as e:
        print(f"❌ LangGraph import failed: {e}")
        return False
    
    try:
        import pytorch_gradcam
        print("✅ PyTorch Grad-CAM imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch Grad-CAM import failed: {e}")
        print("   Note: This is optional for basic functionality")
        return True  # Don't fail the test for this optional dependency
    
    try:
        import google.generativeai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    

    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def test_project_modules():
    """Test if project-specific modules can be imported."""
    print("\n🔍 Testing project modules...")
    
    try:
        from models.image_model import ChestXRayClassifier
        print("✅ Image model imported successfully")
    except ImportError as e:
        print(f"❌ Image model import failed: {e}")
        return False
    
    try:
        from models.grad_cam_tool import XRayGradCAM
        print("✅ Grad-CAM tool imported successfully")
    except ImportError as e:
        print(f"❌ Grad-CAM tool import failed: {e}")
        return False
    

    
    try:
        from agents.symptom_agent import SymptomAgent
        print("✅ Symptom agent imported successfully")
    except ImportError as e:
        print(f"❌ Symptom agent import failed: {e}")
        return False
    
    try:
        from agents.image_agent import ImageAnalysisAgent
        print("✅ Image agent imported successfully")
    except ImportError as e:
        print(f"❌ Image agent import failed: {e}")
        return False
    
    try:
        from agents.triage_agent import TriageAgent
        print("✅ Triage agent imported successfully")
    except ImportError as e:
        print(f"❌ Triage agent import failed: {e}")
        return False
    
    try:
        from agents.report_agent import ReportAgent
        print("✅ Report agent imported successfully")
    except ImportError as e:
        print(f"❌ Report agent import failed: {e}")
        return False
    
    try:
        from workflow import RadiologyWorkflowRunner
        print("✅ Workflow runner imported successfully")
    except ImportError as e:
        print(f"❌ Workflow runner import failed: {e}")
        return False
    
    try:
        from utils.helpers import setup_logging, check_dependencies
        print("✅ Helper utilities imported successfully")
    except ImportError as e:
        print(f"❌ Helper utilities import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration."""
    print("\n🔍 Testing environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (create from env.example)")
    
    # Check Google AI API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and google_key != "your_google_ai_api_key_here":
        print("✅ Google AI API key configured")
    else:
        print("⚠️  Google AI API key not configured (required for LLM features)")
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        print("✅ Data directory exists")
        # Check for sample images
        image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpeg"))
        if image_files:
            print(f"✅ Found {len(image_files)} sample image(s)")
        else:
            print("⚠️  No sample images found in data/ directory")
    else:
        print("⚠️  Data directory not found")
    
    return True

def test_gpu():
    """Test GPU availability."""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  GPU not available (CPU will be used)")
            return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_model_loading():
    """Test if TorchXRayVision model can be loaded."""
    print("\n🔍 Testing model loading...")
    
    try:
        from models.image_model import ChestXRayClassifier
        print("🔄 Loading TorchXRayVision model (this may take a moment)...")
        
        # Initialize classifier (this will download the model if needed)
        classifier = ChestXRayClassifier()
        print("✅ TorchXRayVision model loaded successfully")
        
        # Check available pathologies
        pathologies = classifier.get_key_pathologies()
        print(f"✅ Available pathologies: {len(pathologies)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 AI Radiology Assistant - Setup Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Import Tests", test_imports),
        ("Project Module Tests", test_project_modules),
        ("Environment Tests", test_environment),
        ("GPU Tests", test_gpu),
        ("Model Loading Tests", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The AI Radiology Assistant is ready to use.")
        print("\nNext steps:")
        print("1. Add sample X-ray images to the data/ directory")
        print("2. Configure your Google AI API key in .env file")
        print("3. Run: python main.py --image data/sample.jpg --symptoms 'test symptoms'")
        print("4. Or run: streamlit run ui/app_streamlit.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above and fix them.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check your Python environment and version")
        print("3. Ensure you have sufficient disk space for model downloads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 