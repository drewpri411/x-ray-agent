"""
Image Model Wrapper for TorchXRayVision
Handles loading and inference with pre-trained chest X-ray classifiers.
"""

import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ChestXRayClassifier:
    """Wrapper for TorchXRayVision DenseNet model for chest X-ray analysis."""
    
    def __init__(self, model_name: str = "densenet121-res224-all"):
        """
        Initialize the chest X-ray classifier.
        
        Args:
            model_name: Name of the pre-trained model to load
        """
        self.model_name = model_name
        self.model = None
        self.pathologies = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained TorchXRayVision model."""
        try:
            logger.info(f"Loading TorchXRayVision model: {self.model_name}")
            self.model = xrv.models.DenseNet(weights=self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.pathologies = self.model.pathologies
            logger.info(f"Model loaded successfully. Available pathologies: {len(self.pathologies)}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an X-ray image for model inference.
        
        Args:
            image_path: Path to the X-ray image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image using TorchXRayVision's preprocessing
            img = xrv.datasets.normalize(image_path, 224)
            img = img.unsqueeze(0)  # Add batch dimension
            img = img.to(self.device)
            return img
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """
        Perform inference on an X-ray image.
        
        Args:
            image_path: Path to the X-ray image file
            
        Returns:
            Dictionary mapping pathology names to probability scores
        """
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.sigmoid(outputs[0]).cpu().numpy()
            
            # Create results dictionary
            results = dict(zip(self.pathologies, probabilities))
            
            logger.info(f"Prediction completed for {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to predict on image {image_path}: {e}")
            raise
    
    def get_key_pathologies(self) -> List[str]:
        """Get list of key pathologies we're focusing on."""
        key_pathologies = [
            "Pneumonia",
            "Effusion", 
            "Lung Opacity",
            "Cardiomegaly",
            "Edema",
            "Consolidation"
        ]
        return [p for p in key_pathologies if p in self.pathologies]
    
    def get_prediction_summary(self, image_path: str) -> Dict:
        """
        Get a summary of predictions with key pathologies highlighted.
        
        Args:
            image_path: Path to the X-ray image file
            
        Returns:
            Dictionary with full predictions and key pathology summary
        """
        full_predictions = self.predict(image_path)
        key_pathologies = self.get_key_pathologies()
        
        summary = {
            "full_predictions": full_predictions,
            "key_findings": {k: full_predictions[k] for k in key_pathologies},
            "high_confidence_findings": {
                k: v for k, v in full_predictions.items() 
                if v > 0.5 and k in key_pathologies
            }
        }
        
        return summary 