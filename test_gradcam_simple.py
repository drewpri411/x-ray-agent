#!/usr/bin/env python3
"""
Simple test to verify Grad-CAM is working
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gradcam_simple():
    """Simple test to verify Grad-CAM is working."""
    print("🧪 Simple Grad-CAM Test")
    print("=" * 30)
    
    try:
        from models.grad_cam_tool import XRayGradCAM
        from models.image_model import ChestXRayClassifier
        
        # Initialize
        classifier = ChestXRayClassifier()
        gradcam = XRayGradCAM(classifier.model)
        print("✅ Initialized successfully")
        
        # Test image
        test_image = "data/samples/normal/normal_1.jpg"
        
        if os.path.exists(test_image):
            # Get predictions
            predictions = classifier.predict(test_image)
            key_pathologies = classifier.get_key_pathologies()
            
            print(f"✅ Predictions obtained")
            print(f"✅ Key pathologies: {key_pathologies}")
            
            # Test pathology mapping
            for pathology in key_pathologies[:3]:
                class_index = gradcam._get_pathology_class_index(pathology)
                score = predictions.get(pathology, 0.0)
                print(f"   {pathology} -> class {class_index} (score: {score:.3f})")
            
            # Test heatmap generation without matplotlib
            if key_pathologies:
                top_pathology = key_pathologies[0]
                top_score = predictions.get(top_pathology, 0.0)
                
                if top_score > 0.1:
                    print(f"\n🔍 Testing heatmap generation for {top_pathology}...")
                    
                    # Generate heatmap without matplotlib
                    try:
                        # Load and preprocess image
                        input_tensor = gradcam._prepare_input(test_image)
                        print(f"   ✅ Input tensor shape: {input_tensor.shape}")
                        
                        # Generate heatmap
                        grayscale_cam = gradcam.cam(input_tensor, class_idx=gradcam._get_pathology_class_index(top_pathology))
                        print(f"   ✅ Heatmap generated, type: {type(grayscale_cam)}")
                        
                        # Handle output
                        if isinstance(grayscale_cam, tuple):
                            grayscale_cam = grayscale_cam[0]
                        
                        if hasattr(grayscale_cam, 'cpu'):
                            grayscale_cam = grayscale_cam.cpu().numpy()
                        
                        if grayscale_cam.ndim > 2:
                            grayscale_cam = grayscale_cam[0]
                        
                        print(f"   ✅ Final heatmap shape: {grayscale_cam.shape}")
                        print(f"   ✅ Heatmap min/max: {grayscale_cam.min():.3f}/{grayscale_cam.max():.3f}")
                        
                        # Test overlay generation
                        original_image = gradcam._load_image_for_overlay(test_image)
                        print(f"   ✅ Original image shape: {original_image.shape}")
                        
                        # Convert to tensors for visualize_cam
                        import torch
                        grayscale_cam_tensor = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)
                        original_image_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0)
                        
                        print(f"   ✅ Heatmap tensor shape: {grayscale_cam_tensor.shape}")
                        print(f"   ✅ Image tensor shape: {original_image_tensor.shape}")
                        
                        # Create overlay
                        from gradcam.utils import visualize_cam
                        heatmap, heatmap_image = visualize_cam(grayscale_cam_tensor, original_image_tensor, alpha=0.4)
                        heatmap_image = heatmap_image.permute(1, 2, 0).numpy()
                        
                        print(f"   ✅ Overlay generated successfully!")
                        print(f"   ✅ Overlay shape: {heatmap_image.shape}")
                        
                        print("\n🎉 Grad-CAM is working correctly!")
                        return True
                        
                    except Exception as e:
                        print(f"   ❌ Heatmap generation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    print(f"   ⚠️  Top pathology score too low ({top_score:.3f})")
                    return True
        else:
            print(f"❌ Test image not found: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradcam_simple()
    sys.exit(0 if success else 1) 