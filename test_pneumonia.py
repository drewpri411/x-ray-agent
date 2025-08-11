#!/usr/bin/env python3
"""
Test Pneumonia Image Analysis
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_pneumonia():
    """Test pneumonia image analysis"""
    try:
        from models.image_model import ChestXRayClassifier
        
        # Initialize classifier
        classifier = ChestXRayClassifier()
        
        # Test pneumonia image
        image_path = "data/samples/pneumonia/pneumonia_1.jpg"
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🩺 Analyzing pneumonia image: {image_path}")
        
        # Get predictions
        predictions = classifier.predict(image_path)
        
        # Show results
        print("\n📊 Pneumonia Analysis Results:")
        print("-" * 40)
        
        # Sort by confidence
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        print("Top findings:")
        for i, (pathology, score) in enumerate(sorted_predictions[:5]):
            confidence = "HIGH" if score > 0.7 else "MODERATE" if score > 0.3 else "LOW"
            print(f"  {i+1}. {pathology}: {score:.3f} ({confidence})")
        
        # Check if pneumonia was detected
        pneumonia_score = predictions.get('Pneumonia', 0)
        print(f"\n🔍 Pneumonia Detection: {pneumonia_score:.3f}")
        
        if pneumonia_score > 0.5:
            print("✅ Pneumonia detected with high confidence!")
        elif pneumonia_score > 0.3:
            print("⚠️  Possible pneumonia detected")
        else:
            print("❌ No significant pneumonia detected")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_pneumonia() 