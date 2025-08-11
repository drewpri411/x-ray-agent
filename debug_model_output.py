#!/usr/bin/env python3
"""
Debug script to check model output format
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_model_output():
    """Debug the model output format."""
    print("🔍 Debugging Model Output")
    print("=" * 30)
    
    try:
        from models.image_model import ChestXRayClassifier
        
        # Initialize classifier
        classifier = ChestXRayClassifier()
        print("✅ Classifier initialized")
        
        # Test image
        test_image = "data/samples/normal/normal_1.jpg"
        
        if os.path.exists(test_image):
            print(f"✅ Test image found: {test_image}")
            
            # Preprocess image
            img_tensor = classifier.preprocess_image(test_image)
            print(f"✅ Image preprocessed, tensor shape: {img_tensor.shape}")
            
            # Get model output
            import torch
            with torch.no_grad():
                outputs = classifier.model(img_tensor)
                print(f"✅ Model output type: {type(outputs)}")
                print(f"✅ Model output: {outputs}")
                
                if isinstance(outputs, tuple):
                    print(f"✅ Output is tuple with {len(outputs)} elements")
                    for i, output in enumerate(outputs):
                        print(f"   Element {i}: type={type(output)}, shape={output.shape if hasattr(output, 'shape') else 'N/A'}")
                elif isinstance(outputs, dict):
                    print(f"✅ Output is dict with keys: {list(outputs.keys())}")
                    for key, value in outputs.items():
                        print(f"   Key '{key}': type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
                else:
                    print(f"✅ Output is {type(outputs)} with shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
        
        else:
            print(f"❌ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_output() 