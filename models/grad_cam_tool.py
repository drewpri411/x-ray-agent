"""
Grad-CAM Tool for X-Ray Explainability
Generates heatmaps to visualize model attention on chest X-ray images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
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
                self.target_layers = [model.features.denseblock4]
            else:
                # Fallback: find the last convolutional layer
                self.target_layers = self._find_last_conv_layer(model)
        
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=torch.cuda.is_available())
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
            
            # Create target
            targets = [ClassifierOutputTarget(target_class)]
            
            # Generate heatmap
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Load original image for overlay
            original_image = self._load_image_for_overlay(image_path)
            
            # Create heatmap overlay
            heatmap_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title('Original X-Ray')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(grayscale_cam, cmap='jet')
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
        Map pathology name to class index.
        This is a simplified mapping - in practice, you'd need the exact class mapping
        from your TorchXRayVision model.
        """
        # Common pathology mappings (this should match your model's class mapping)
        pathology_mapping = {
            'pneumonia': 0,
            'pleural_effusion': 1,
            'atelectasis': 2,
            'cardiomegaly': 3,
            'consolidation': 4,
            'edema': 5,
            'fracture': 6,
            'mass': 7,
            'nodule': 8,
            'pneumothorax': 9
        }
        
        # Try exact match first
        if pathology_name in pathology_mapping:
            return pathology_mapping[pathology_name]
        
        # Try partial matches
        for key, value in pathology_mapping.items():
            if key in pathology_name.lower() or pathology_name.lower() in key:
                return value
        
        # Default to first class if no match found
        logger.warning(f"No class mapping found for pathology: {pathology_name}")
        return 0
    
    def _prepare_input(self, image_path: str) -> torch.Tensor:
        """Prepare input tensor for the model."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to model input size (typically 224x224 for DenseNet)
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _load_image_for_overlay(self, image_path: str) -> np.ndarray:
        """Load image for heatmap overlay."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        return np.array(image) / 255.0 