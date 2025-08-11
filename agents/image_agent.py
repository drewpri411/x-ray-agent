"""
Image Agent for X-ray Analysis
Coordinates image preprocessing, model inference, and result formatting.
"""

from typing import Dict, Any, Optional
from models.image_model import ChestXRayClassifier
from models.grad_cam_tool import XRayGradCAM
import logging
import os

logger = logging.getLogger(__name__)

class ImageAnalysisAgent:
    """Agent for analyzing chest X-ray images using TorchXRayVision."""
    
    def __init__(self, model_name: str = "densenet121-res224-all"):
        """
        Initialize the image analysis agent.
        
        Args:
            model_name: Name of the TorchXRayVision model to use
        """
        self.classifier = ChestXRayClassifier(model_name)
        
        # Initialize Grad-CAM if available
        try:
            self.grad_cam = XRayGradCAM(self.classifier.model)
            self.grad_cam_available = True
            logger.info("Image analysis agent initialized with Grad-CAM support")
        except Exception as e:
            self.grad_cam = None
            self.grad_cam_available = False
            logger.warning(f"Grad-CAM not available: {e}. Continuing without heatmap generation.")
        
        logger.info("Image analysis agent initialized successfully")
    
    def analyze_image(self, image_path: str, generate_heatmaps: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive image analysis including classification and explainability.
        
        Args:
            image_path: Path to the X-ray image file
            generate_heatmaps: Whether to generate Grad-CAM heatmaps
            
        Returns:
            Dictionary containing analysis results (classification, heatmaps, findings)
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Perform classification
            logger.info(f"Starting image analysis for: {image_path}")
            prediction_summary = self.classifier.get_prediction_summary(image_path)
            
            # Generate heatmaps if requested
            heatmaps = {}
            if generate_heatmaps:
                heatmaps = self._generate_heatmaps(image_path, prediction_summary['full_predictions'])
            
            # Compile results
            analysis_results = {
                'image_path': image_path,
                'classification_results': prediction_summary,
                'heatmaps': heatmaps,
                'key_findings': prediction_summary['key_findings'],
                'high_confidence_findings': prediction_summary['high_confidence_findings'],
                'analysis_status': 'completed'
            }
            
            logger.info(f"Image analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            return {
                'image_path': image_path,
                'classification_results': {},
                'heatmaps': {},
                'key_findings': {},
                'high_confidence_findings': {},
                'analysis_status': 'failed',
                'error': str(e)
            }
    

    
    def get_urgent_findings(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract urgent findings from analysis results.
        
        Args:
            analysis_results: Results from image analysis
            
        Returns:
            Dictionary containing urgent findings
        """
        urgent_findings = {
            'has_urgent_findings': False,
            'urgent_pathologies': [],
            'urgency_score': 0.0,
            'recommendations': [],
            'key_findings': {}  # Add key_findings to the structure
        }
        
        try:
            key_findings = analysis_results.get('key_findings', {})
            high_confidence = analysis_results.get('high_confidence_findings', {})
            
            # Define urgent pathologies and their thresholds
            urgent_pathologies = {
                'Pneumonia': 0.6,  # Lowered threshold
                'Effusion': 0.6,   # Lowered threshold
                'Lung Opacity': 0.6,  # Lowered threshold
                'Cardiomegaly': 0.6,  # Lowered threshold
                'Edema': 0.6        # Lowered threshold
            }
            
            # Check for urgent findings
            for pathology, threshold in urgent_pathologies.items():
                if pathology in key_findings:
                    score = key_findings[pathology]
                    if score >= threshold:
                        urgent_findings['urgent_pathologies'].append({
                            'pathology': pathology,
                            'score': score,
                            'threshold': threshold
                        })
            
            # Store key_findings for triage assessment
            urgent_findings['key_findings'] = key_findings
            
            # Determine overall urgency
            if urgent_findings['urgent_pathologies']:
                urgent_findings['has_urgent_findings'] = True
                # Calculate overall urgency score
                max_score = max([f['score'] for f in urgent_findings['urgent_pathologies']])
                urgent_findings['urgency_score'] = max_score
                
                # Generate recommendations
                urgent_findings['recommendations'] = self._generate_urgent_recommendations(
                    urgent_findings['urgent_pathologies']
                )
            else:
                # Even if no urgent pathologies, consider moderate findings
                if key_findings:
                    max_score = max(key_findings.values())
                    if max_score >= 0.5:  # Moderate threshold
                        urgent_findings['urgency_score'] = max_score * 0.7  # Scale down for moderate findings
            
            return urgent_findings
            
        except Exception as e:
            logger.error(f"Failed to extract urgent findings: {e}")
            return urgent_findings
    
    def _generate_urgent_recommendations(self, urgent_pathologies: list) -> list:
        """Generate recommendations for urgent findings."""
        recommendations = []
        
        for finding in urgent_pathologies:
            pathology = finding['pathology']
            score = finding['score']
            
            if pathology == 'Pneumonia':
                if score > 0.8:
                    recommendations.append("URGENT: High probability of pneumonia - immediate medical attention required")
                else:
                    recommendations.append("MODERATE: Possible pneumonia - prompt evaluation recommended")
            
            elif pathology == 'Effusion':
                recommendations.append("URGENT: Pleural effusion detected - requires immediate evaluation")
            
            elif pathology == 'Lung Opacity':
                recommendations.append("MODERATE: Lung opacity detected - further evaluation needed")
            
            elif pathology == 'Cardiomegaly':
                recommendations.append("URGENT: Cardiomegaly detected - cardiac evaluation required")
            
            elif pathology == 'Edema':
                recommendations.append("URGENT: Pulmonary edema detected - immediate attention required")
        
        return recommendations
    
    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate that the image is suitable for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'is_valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check file exists
            if not os.path.exists(image_path):
                validation_result['errors'].append("Image file not found")
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size < 1000:  # Less than 1KB
                validation_result['errors'].append("Image file too small")
            elif file_size > 50 * 1024 * 1024:  # More than 50MB
                validation_result['warnings'].append("Image file very large")
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in valid_extensions:
                validation_result['errors'].append(f"Unsupported file format: {file_ext}")
            
            # If no errors, mark as valid
            if not validation_result['errors']:
                validation_result['is_valid'] = True
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
            
    def _generate_heatmaps(self, image_path: str, pathology_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmaps for top pathologies.
        
        Args:
            image_path: Path to the X-ray image
            pathology_scores: Dictionary of pathology scores
            
        Returns:
            Dictionary containing heatmap data
        """
        # Check if Grad-CAM is available
        if not self.grad_cam_available or self.grad_cam is None:
            logger.info("Grad-CAM not available - skipping heatmap generation")
            return {}
        
        try:
            # Generate heatmaps for top 3 pathologies with scores > 0.1
            heatmaps = self.grad_cam.generate_multi_pathology_heatmaps(
                image_path, pathology_scores, top_k=3
            )
            
            logger.info(f"Generated {len(heatmaps)} heatmaps")
            return heatmaps
            
        except Exception as e:
            logger.error(f"Failed to generate heatmaps: {e}")
            return {} 