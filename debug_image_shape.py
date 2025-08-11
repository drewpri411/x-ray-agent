#!/usr/bin/env python3
"""
Debug script to check image shapes
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_image_shape():
    """Debug image shapes."""
    print("🔍 Debugging Image Shapes")
    print("=" * 30)
    
    try:
        test_image = "data/samples/normal/normal_1.jpg"
        
        if os.path.exists(test_image):
            from PIL import Image
            import numpy as np
            
            # Load image
            image = Image.open(test_image)
            print(f"✅ Original image mode: {image.mode}")
            print(f"✅ Original image size: {image.size}")
            
            # Convert to RGB
            image_rgb = image.convert('RGB')
            print(f"✅ RGB image mode: {image_rgb.mode}")
            print(f"✅ RGB image size: {image_rgb.size}")
            
            # Resize
            image_resized = image_rgb.resize((224, 224))
            print(f"✅ Resized image size: {image_resized.size}")
            
            # Convert to numpy
            image_array = np.array(image_resized)
            print(f"✅ Numpy array shape: {image_array.shape}")
            print(f"✅ Numpy array dtype: {image_array.dtype}")
            
            # Normalize
            image_normalized = image_array / 255.0
            print(f"✅ Normalized array shape: {image_normalized.shape}")
            print(f"✅ Normalized array dtype: {image_normalized.dtype}")
            print(f"✅ Normalized array min/max: {image_normalized.min():.3f}/{image_normalized.max():.3f}")
            
            # Convert to tensor
            import torch
            image_tensor = torch.from_numpy(image_normalized).float()
            print(f"✅ Tensor shape: {image_tensor.shape}")
            
            # Permute for CHW format
            image_tensor_chw = image_tensor.permute(2, 0, 1)
            print(f"✅ CHW tensor shape: {image_tensor_chw.shape}")
            
            # Add batch dimension
            image_tensor_bchw = image_tensor_chw.unsqueeze(0)
            print(f"✅ BCHW tensor shape: {image_tensor_bchw.shape}")
            
        else:
            print(f"❌ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_image_shape() 