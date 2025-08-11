"""
Grad-CAM Tool for X-Ray Explainability
Generates heatmaps to visualize model attention on chest X-ray images.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

try:
    from gradcam import GradCAM
    from gradcam.utils import visualize_cam
    GRADCAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch Grad-CAM not available: {e}")
    GRADCAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class XRayGradCAM:
    """Grad-CAM implementation for chest X-ray explainability."""
    
    def __init__(self, model, target_layers=None):
        """
        Initialize Grad-CAM with the model.
        
        Args:
            model: PyTorch model for X-ray classification
            target_layers: List of target layers for Grad-CAM (defaults to last conv layer)
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError("PyTorch Grad-CAM is required but not available")
            
        self.model = model
        self.model.eval()
        
        # Set target layers if not provided
        if target_layers is None:
            # For DenseNet, typically the last convolutional layer
            if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
                # Get the last layer of denseblock4
                denseblock4 = model.features.denseblock4
                if hasattr(denseblock4, 'denselayer32'):
                    self.target_layers = [denseblock4.denselayer32]
                else:
                    # Get the last dense layer in denseblock4
                    last_layer_name = f'denselayer{len(denseblock4)}'
                    if hasattr(denseblock4, last_layer_name):
                        self.target_layers = [getattr(denseblock4, last_layer_name)]
                    else:
                        self.target_layers = [denseblock4]
            else:
                # Fallback: find the last convolutional layer
                self.target_layers = self._find_last_conv_layer(model)
        
        self.cam = GradCAM(arch=self.model, target_layer=self.target_layers[0])
        logger.info("Grad-CAM initialized successfully")
    
    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        
        if conv_layers:
            return [conv_layers[-1]]
        else:
            raise ValueError("No convolutional layers found in model")
    
    def generate_heatmap(self, image_path: str, target_class: int, 
                        target_score: float = None) -> Dict[str, Any]:
        """
        Generate a single Grad-CAM heatmap for a specific pathology.
        
        Args:
            image_path: Path to the X-ray image
            target_class: Class index for the pathology
            target_score: Confidence score for the pathology
            
        Returns:
            Dictionary containing heatmap data and visualization
        """
        try:
            # Load and preprocess image
            input_tensor = self._prepare_input(image_path)
            
            # Generate heatmap using the gradcam package
            grayscale_cam = self.cam(input_tensor, class_idx=target_class)
            # Handle different output formats
            if isinstance(grayscale_cam, tuple):
                grayscale_cam = grayscale_cam[0]
            
            # Convert to numpy if it's a tensor
            if hasattr(grayscale_cam, 'cpu'):
                grayscale_cam = grayscale_cam.cpu().numpy()
            
            # Get the first batch if needed
            if grayscale_cam.ndim > 2:
                grayscale_cam = grayscale_cam[0]
            
            # Load original image for overlay
            original_image = self._load_image_for_overlay(image_path)
            
            # Convert to tensors for visualize_cam
            import torch
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            original_image_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
            # Create heatmap overlay using visualize_cam
            heatmap, heatmap_image = visualize_cam(grayscale_cam_tensor, original_image_tensor, alpha=0.4)
            heatmap_image = heatmap_image.permute(1, 2, 0).numpy()  # Convert back to numpy
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title('Original X-Ray')
            axes[0].axis('off')
            
            # Heatmap
            if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
                grayscale_cam_2d = grayscale_cam[0]
            else:
                grayscale_cam_2d = grayscale_cam
            axes[1].imshow(grayscale_cam_2d, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(heatmap_image)
            axes[2].set_title('Heatmap Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            return {
                'target_class': target_class,
                'target_score': target_score,
                'heatmap': grayscale_cam,
                'overlay': heatmap_image,
                'figure': fig,
                'original_image': original_image
            }
            
        except Exception as e:
            logger.error(f"Failed to generate heatmap for {image_path}: {e}")
            return {
                'target_class': target_class,
                'target_score': target_score,
                'error': str(e)
            }
    
    def generate_multi_pathology_heatmaps(self, image_path: str, 
                                        pathology_scores: Dict[str, float], 
                                        top_k: int = 3) -> Dict[str, Any]:
        """
        Generate heatmaps for multiple pathologies.
        
        Args:
            image_path: Path to the X-ray image
            pathology_scores: Dictionary mapping pathology names to scores
            top_k: Number of top pathologies to generate heatmaps for
            
        Returns:
            Dictionary mapping pathology names to heatmap data
        """
        # Sort pathologies by score and get top k
        sorted_pathologies = sorted(pathology_scores.items(), 
                                  key=lambda x: x[1], reverse=True)
        top_pathologies = sorted_pathologies[:top_k]
        
        heatmaps = {}
        
        for pathology_name, score in top_pathologies:
            if score > 0.1:  # Only generate heatmaps for significant findings
                # Map pathology name to class index (this would need to be customized)
                target_class = self._get_pathology_class_index(pathology_name)
                
                heatmap_data = self.generate_heatmap(
                    image_path, target_class, score
                )
                heatmaps[pathology_name] = heatmap_data
        
        return heatmaps
    
    def _get_pathology_class_index(self, pathology_name: str) -> int:
        """
        Map pathology name to class index using the model's pathology list.
        """
        if not hasattr(self.model, 'pathologies'):
            logger.warning("Model doesn't have pathologies attribute")
            return 0
        
        # Get the model's pathology list
        model_pathologies = self.model.pathologies
        
        # Try exact match first
        if pathology_name in model_pathologies:
            return model_pathologies.index(pathology_name)
        
        # Try case-insensitive match
        for i, path in enumerate(model_pathologies):
            if path.lower() == pathology_name.lower():
                return i
        
        # Try partial matches
        for i, path in enumerate(model_pathologies):
            if (pathology_name.lower() in path.lower() or 
                path.lower() in pathology_name.lower()):
                return i
        
        # Default to first class if no match found
        logger.warning(f"No class mapping found for pathology: {pathology_name}")
        logger.info(f"Available pathologies: {model_pathologies[:10]}...")  # Show first 10
        return 0
    
    def _prepare_input(self, image_path: str) -> torch.Tensor:
        """Prepare input tensor for the model using TorchXRayVision preprocessing."""
        try:
            # Load image with PIL first
            from PIL import Image
            import numpy as np
            
            # Load image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((224, 224))
            img_array = np.array(img)
            
            # Use TorchXRayVision's normalize function
            import torchxrayvision as xrv
            img_normalized = xrv.datasets.normalize(img_array, 255)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_normalized).float()
            
            # Add channel and batch dimensions if needed
            if img_tensor.dim() == 2:  # [H, W]
                img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]
            if img_tensor.dim() == 3:  # [C, H, W]
                img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image for Grad-CAM: {e}")
            # Fallback to basic preprocessing
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.from_numpy(np.array(image)).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor
    
    def _load_image_for_overlay(self, image_path: str) -> np.ndarray:
        """Load image for heatmap overlay."""
        image = Image.open(image_path).convert('RGB')  # Convert to RGB for overlay
        image = image.resize((224, 224))
        return np.array(image) / 255.0 