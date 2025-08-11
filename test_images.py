#!/usr/bin/env python3
"""
Simple Test Script for AI Radiology Assistant
Tests image analysis without Grad-CAM dependencies
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_image_analysis():
    """Test image analysis with the downloaded images"""
    print("🩺 Testing AI Radiology Assistant with Your Images")
    print("=" * 60)
    
    try:
        from models.image_model import ChestXRayClassifier
        
        # Initialize the classifier
        print("Loading TorchXRayVision model...")
        classifier = ChestXRayClassifier()
        print("✅ Model loaded successfully")
        
        # Test with normal image
        image_path = "data/samples/normal/normal_1.jpg"
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"Analyzing image: {image_path}")
        
        # Get predictions
        predictions = classifier.predict(image_path)
        
        print("\n📊 Analysis Results:")
        print("-" * 40)
        
        # Show key pathologies
        key_pathologies = classifier.get_key_pathologies()
        print(f"Available pathologies: {len(key_pathologies)}")
        print(f"Pathologies: {', '.join(key_pathologies)}")
        
        # Show top predictions
        print("\nTop findings:")
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        for i, (pathology, score) in enumerate(sorted_predictions[:5]):
            confidence = "HIGH" if score > 0.7 else "MODERATE" if score > 0.3 else "LOW"
            print(f"  {i+1}. {pathology}: {score:.3f} ({confidence})")
        
        # Get summary
        summary = classifier.get_prediction_summary(image_path)
        print(f"\nSummary: {summary}")
        
        print("\n✅ Image analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def test_multiple_images():
    """Test multiple images from different categories"""
    print("\n🔄 Testing Multiple Image Categories")
    print("=" * 40)
    
    try:
        from models.image_model import ChestXRayClassifier
        
        classifier = ChestXRayClassifier()
        
        # Test categories
        categories = [
            ("normal", "data/samples/normal/normal_1.jpg"),
            ("pneumonia", "data/samples/pneumonia/pneumonia_1.jpg"),
            ("pleural_effusion", "data/samples/pleural_effusion/pleural_1.jpg"),
            ("atelectasis", "data/samples/atelectasis/atelectasis_1.jpg"),
            ("cardiomegaly", "data/samples/cardiomegaly/cardiomegaly_1.jpg")
        ]
        
        for category, image_path in categories:
            if os.path.exists(image_path):
                print(f"\n📸 Testing {category} image...")
                predictions = classifier.predict(image_path)
                
                # Show top finding
                if predictions:
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    top_pathology, top_score = sorted_predictions[0]
                    confidence = "HIGH" if top_score > 0.7 else "MODERATE" if top_score > 0.3 else "LOW"
                    print(f"  Top finding: {top_pathology} ({top_score:.3f}) - {confidence}")
                else:
                    print(f"  No predictions generated")
            else:
                print(f"  ⚠️  Image not found: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multiple images: {e}")
        return False

def test_web_interface():
    """Test if web interface is accessible"""
    print("\n🌐 Web Interface Status:")
    print("-" * 30)
    
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit web interface is running")
            print("🌐 Access at: http://localhost:8501")
            return True
        else:
            print("⚠️  Web interface may not be running")
            return False
    except:
        print("⚠️  Web interface not accessible")
        print("💡 Start with: streamlit run ui/app_streamlit.py")
        return False

def main():
    """Main test function"""
    print("🧪 AI Radiology Assistant - Image Testing")
    print("=" * 60)
    
    # Test single image analysis
    analysis_success = test_image_analysis()
    
    # Test multiple images
    multiple_success = test_multiple_images()
    
    # Test web interface
    web_success = test_web_interface()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Single Image Analysis: {'✅ PASS' if analysis_success else '❌ FAIL'}")
    print(f"  Multiple Image Testing: {'✅ PASS' if multiple_success else '❌ FAIL'}")
    print(f"  Web Interface: {'✅ PASS' if web_success else '⚠️  CHECK'}")
    
    if analysis_success:
        print("\n🎉 Core functionality is working!")
        print("\n📋 Next steps:")
        print("1. Use web interface for interactive testing")
        print("2. Test with different pathologies")
        print("3. Configure API key for full LLM features")
        
        print("\n🌐 Web Interface:")
        print("  - Go to: http://localhost:8501")
        print("  - Upload your images")
        print("  - Enter symptoms and analyze")
        
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return 0 if analysis_success else 1

if __name__ == "__main__":
    sys.exit(main()) 