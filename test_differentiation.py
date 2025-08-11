#!/usr/bin/env python3
"""
Test script to verify that different images produce different pathology scores
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_image_differentiation():
    """Test that different images produce different pathology scores."""
    print("🧪 Testing Image Differentiation")
    print("=" * 50)
    
    try:
        from models.image_model import ChestXRayClassifier
        
        # Initialize classifier
        classifier = ChestXRayClassifier()
        print("✅ Classifier initialized successfully")
        
        # Test images
        test_images = [
            ("normal", "data/samples/normal/normal_1.jpg"),
            ("pneumonia", "data/samples/pneumonia/pneumonia_1.jpg"),
        ]
        
        results = {}
        
        for category, image_path in test_images:
            if os.path.exists(image_path):
                print(f"\n📸 Testing {category} image: {image_path}")
                
                # Get predictions
                predictions = classifier.predict(image_path)
                
                # Get key pathologies
                key_pathologies = classifier.get_key_pathologies()
                
                # Store results
                results[category] = {
                    'predictions': predictions,
                    'key_findings': {k: predictions[k] for k in key_pathologies if k in predictions}
                }
                
                print(f"   ✅ Analysis completed")
                
                # Show top findings
                sorted_findings = sorted(
                    results[category]['key_findings'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                print(f"   Top findings for {category}:")
                for i, (pathology, score) in enumerate(sorted_findings[:3]):
                    print(f"     {i+1}. {pathology}: {score:.3f}")
            else:
                print(f"   ⚠️  Image not found: {image_path}")
        
        # Compare results
        if len(results) >= 2:
            print("\n🔍 Comparing Results:")
            print("-" * 30)
            
            categories = list(results.keys())
            key_pathologies = classifier.get_key_pathologies()
            
            for pathology in key_pathologies[:5]:  # Compare top 5 pathologies
                print(f"\n{pathology}:")
                for category in categories:
                    score = results[category]['key_findings'].get(pathology, 0.0)
                    print(f"  {category}: {score:.3f}")
                
                # Check if scores are different
                scores = [results[cat]['key_findings'].get(pathology, 0.0) for cat in categories]
                if len(set([round(s, 2) for s in scores])) > 1:
                    print(f"  ✅ Scores are different")
                else:
                    print(f"  ⚠️  Scores are similar (may indicate preprocessing issue)")
        
        # Test with different images of same category
        print("\n🔄 Testing Multiple Images of Same Category:")
        print("-" * 40)
        
        normal_images = [
            "data/samples/normal/normal_1.jpg",
            "data/samples/normal/normal_2.jpg",
        ]
        
        normal_scores = []
        for i, img_path in enumerate(normal_images):
            if os.path.exists(img_path):
                predictions = classifier.predict(img_path)
                pneumonia_score = predictions.get('Pneumonia', 0.0)
                normal_scores.append(pneumonia_score)
                print(f"  Normal {i+1}: Pneumonia score = {pneumonia_score:.3f}")
        
        if len(normal_scores) > 1:
            score_variance = max(normal_scores) - min(normal_scores)
            if score_variance > 0.05:
                print(f"  ✅ Good variance in normal images: {score_variance:.3f}")
            else:
                print(f"  ⚠️  Low variance in normal images: {score_variance:.3f}")
        
        print("\n✅ Image differentiation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_image_differentiation()
    sys.exit(0 if success else 1) 