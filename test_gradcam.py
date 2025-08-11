#!/usr/bin/env python3
"""
Test script to verify Grad-CAM functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gradcam():
    """Test Grad-CAM functionality."""
    print("🧪 Testing Grad-CAM Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Check if pytorch-gradcam is available
        print("1. Checking Grad-CAM Dependencies...")
        try:
            import gradcam
            print("   ✅ pytorch-gradcam is available")
        except ImportError as e:
            print(f"   ❌ pytorch-gradcam not available: {e}")
            print("   💡 Install with: pip install pytorch-gradcam")
            return False
        
        # Test 2: Import our Grad-CAM tool
        print("\n2. Testing Grad-CAM Tool Import...")
        try:
            from models.grad_cam_tool import XRayGradCAM, GRADCAM_AVAILABLE
            print(f"   ✅ Grad-CAM tool imported successfully")
            print(f"   ✅ GRADCAM_AVAILABLE: {GRADCAM_AVAILABLE}")
        except Exception as e:
            print(f"   ❌ Failed to import Grad-CAM tool: {e}")
            return False
        
        # Test 3: Initialize image classifier
        print("\n3. Initializing Image Classifier...")
        try:
            from models.image_model import ChestXRayClassifier
            classifier = ChestXRayClassifier()
            print("   ✅ Image classifier initialized successfully")
            print(f"   ✅ Available pathologies: {len(classifier.pathologies)}")
            print(f"   ✅ First 5 pathologies: {classifier.pathologies[:5]}")
        except Exception as e:
            print(f"   ❌ Failed to initialize classifier: {e}")
            return False
        
        # Test 4: Initialize Grad-CAM
        print("\n4. Initializing Grad-CAM...")
        try:
            gradcam = XRayGradCAM(classifier.model)
            print("   ✅ Grad-CAM initialized successfully")
            print(f"   ✅ Target layers: {len(gradcam.target_layers)}")
        except Exception as e:
            print(f"   ❌ Failed to initialize Grad-CAM: {e}")
            return False
        
        # Test 5: Test with a sample image
        print("\n5. Testing Grad-CAM with Sample Image...")
        test_image = "data/samples/normal/normal_1.jpg"
        
        if os.path.exists(test_image):
            try:
                # Get predictions first
                predictions = classifier.predict(test_image)
                key_pathologies = classifier.get_key_pathologies()
                
                print(f"   ✅ Image loaded: {test_image}")
                print(f"   ✅ Key pathologies: {key_pathologies}")
                
                # Test pathology mapping
                print("\n   Testing Pathology Mapping:")
                for pathology in key_pathologies[:3]:  # Test first 3
                    class_index = gradcam._get_pathology_class_index(pathology)
                    score = predictions.get(pathology, 0.0)
                    print(f"     {pathology} -> class {class_index} (score: {score:.3f})")
                
                # Test heatmap generation for top pathology
                if key_pathologies:
                    top_pathology = key_pathologies[0]
                    top_score = predictions.get(top_pathology, 0.0)
                    
                    if top_score > 0.1:  # Only test if score is significant
                        print(f"\n   Generating heatmap for {top_pathology}...")
                        heatmap_data = gradcam.generate_heatmap(
                            test_image, 
                            gradcam._get_pathology_class_index(top_pathology),
                            top_score
                        )
                        
                        if 'error' not in heatmap_data:
                            print("   ✅ Heatmap generated successfully")
                            print(f"   ✅ Heatmap shape: {heatmap_data['heatmap'].shape}")
                            print(f"   ✅ Overlay shape: {heatmap_data['overlay'].shape}")
                        else:
                            print(f"   ❌ Heatmap generation failed: {heatmap_data['error']}")
                    else:
                        print(f"   ⚠️  Top pathology score too low ({top_score:.3f}) for heatmap test")
                
                # Test multi-pathology heatmaps
                print(f"\n   Testing Multi-Pathology Heatmaps...")
                heatmaps = gradcam.generate_multi_pathology_heatmaps(
                    test_image, predictions, top_k=2
                )
                
                if heatmaps:
                    print(f"   ✅ Generated {len(heatmaps)} heatmaps")
                    for pathology, data in heatmaps.items():
                        if 'error' not in data:
                            print(f"     ✅ {pathology}: heatmap generated")
                        else:
                            print(f"     ❌ {pathology}: {data['error']}")
                else:
                    print("   ⚠️  No heatmaps generated (scores may be too low)")
                
            except Exception as e:
                print(f"   ❌ Failed to test with image: {e}")
                return False
        else:
            print(f"   ⚠️  Test image not found: {test_image}")
        
        print("\n✅ Grad-CAM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gradcam()
    sys.exit(0 if success else 1) 