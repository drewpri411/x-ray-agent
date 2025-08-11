"""
Triage Agent for Automated Urgency Assessment
Combines image analysis results and symptom data to determine case urgency.
"""

from typing import Dict, Any, List
import logging
from agents.symptom_agent import SymptomAgent

logger = logging.getLogger(__name__)

class TriageAgent:
    """Agent for automated triage and urgency assessment."""
    
    def __init__(self):
        """Initialize the triage agent."""
        self.symptom_agent = SymptomAgent()
        logger.info("Triage agent initialized successfully")
    
    def assess_urgency(self, 
                      image_analysis: Dict[str, Any], 
                      symptoms: str = "",
                      parsed_symptoms: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess overall case urgency based on image findings and symptoms.
        
        Args:
            image_analysis: Results from image analysis
            symptoms: Raw symptom text (if parsed_symptoms not provided)
            parsed_symptoms: Pre-parsed symptom information
            
        Returns:
            Dictionary containing triage assessment
        """
        try:
            # Parse symptoms if not provided
            if parsed_symptoms is None:
                parsed_symptoms = self.symptom_agent.parse_symptoms(symptoms)
            
            # Extract urgent findings from image analysis
            urgent_findings = image_analysis.get('urgent_findings', {})
            if not urgent_findings:
                # If not already extracted, extract them
                from agents.image_agent import ImageAnalysisAgent
                img_agent = ImageAnalysisAgent()
                urgent_findings = img_agent.get_urgent_findings(image_analysis)
            
            # Calculate combined urgency score
            combined_urgency = self._calculate_combined_urgency(
                urgent_findings, parsed_symptoms
            )
            
            # Determine triage category
            triage_category = self._determine_triage_category(combined_urgency)
            
            # Generate triage recommendations
            recommendations = self._generate_triage_recommendations(
                triage_category, urgent_findings, parsed_symptoms
            )
            
            triage_result = {
                'triage_category': triage_category,
                'urgency_score': combined_urgency,
                'image_urgency': urgent_findings.get('urgency_score', 0.0),
                'symptom_urgency': self.symptom_agent.get_urgency_score(parsed_symptoms),
                'urgent_findings': urgent_findings,
                'parsed_symptoms': parsed_symptoms,
                'recommendations': recommendations,
                'assessment_timestamp': self._get_timestamp()
            }
            
            logger.info(f"Triage assessment completed: {triage_category} (score: {combined_urgency:.3f})")
            return triage_result
            
        except Exception as e:
            logger.error(f"Failed to assess urgency: {e}")
            return {
                'triage_category': 'ERROR',
                'urgency_score': 0.0,
                'image_urgency': 0.0,
                'symptom_urgency': 0.0,
                'urgent_findings': {},
                'parsed_symptoms': {},
                'recommendations': ['Error in triage assessment'],
                'error': str(e)
            }
    
    def _calculate_combined_urgency(self, 
                                  urgent_findings: Dict[str, Any], 
                                  parsed_symptoms: Dict[str, Any]) -> float:
        """
        Calculate combined urgency score from image and symptom data.
        
        Args:
            urgent_findings: Urgent findings from image analysis
            parsed_symptoms: Parsed symptom information
            
        Returns:
            Combined urgency score between 0 and 1
        """
        # Get individual urgency scores
        image_urgency = urgent_findings.get('urgency_score', 0.0)
        symptom_urgency = self.symptom_agent.get_urgency_score(parsed_symptoms)
        
        # Calculate base urgency from image findings
        # Consider moderate findings even if they don't meet urgent threshold
        key_findings = urgent_findings.get('key_findings', {})
        if key_findings:
            # Get the highest pathology score as base urgency
            max_pathology_score = max(key_findings.values())
            # Use a combination of urgent findings and highest pathology score
            image_urgency = max(image_urgency, max_pathology_score * 0.8)
        
        # Weight the scores (image findings may be more critical)
        image_weight = 0.7
        symptom_weight = 0.3
        
        # Calculate weighted average
        combined_score = (image_urgency * image_weight) + (symptom_urgency * symptom_weight)
        
        # Apply additional factors
        combined_score = self._apply_urgency_factors(combined_score, urgent_findings, parsed_symptoms)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, combined_score))
    
    def _apply_urgency_factors(self, 
                              base_score: float, 
                              urgent_findings: Dict[str, Any], 
                              parsed_symptoms: Dict[str, Any]) -> float:
        """
        Apply additional urgency factors to the base score.
        
        Args:
            base_score: Base urgency score
            urgent_findings: Urgent findings from image analysis
            parsed_symptoms: Parsed symptom information
            
        Returns:
            Adjusted urgency score
        """
        adjusted_score = base_score
        
        # Factor 1: Multiple urgent findings increase urgency
        num_urgent_findings = len(urgent_findings.get('urgent_pathologies', []))
        if num_urgent_findings > 1:
            adjusted_score += 0.05 * (num_urgent_findings - 1)  # Reduced from 0.1
        
        # Factor 2: Emergency symptoms increase urgency
        if parsed_symptoms.get('severity_classification') == 'emergency':
            adjusted_score += 0.1  # Reduced from 0.2
        
        # Factor 3: High fever with respiratory symptoms
        symptoms_text = ' '.join(parsed_symptoms.get('primary_symptoms', [])).lower()
        if 'fever' in symptoms_text and any(s in symptoms_text for s in ['cough', 'breathing', 'chest']):
            adjusted_score += 0.08  # Reduced from 0.15
        
        # Factor 4: Duration of symptoms (longer duration may indicate chronic condition)
        duration = parsed_symptoms.get('duration', '').lower()
        if 'week' in duration or 'month' in duration:
            adjusted_score += 0.02  # Reduced from 0.05
        
        return adjusted_score
    
    def _determine_triage_category(self, urgency_score: float) -> str:
        """
        Determine triage category based on urgency score.
        
        Args:
            urgency_score: Combined urgency score
            
        Returns:
            Triage category string
        """
        if urgency_score >= 0.8:  # Lowered threshold for emergency
            return 'EMERGENCY'
        elif urgency_score >= 0.6:  # Lowered threshold for urgent
            return 'URGENT'
        elif urgency_score >= 0.4:  # Lowered threshold for moderate
            return 'MODERATE'
        elif urgency_score >= 0.2:  # Lowered threshold for routine
            return 'ROUTINE'
        else:
            return 'NORMAL'
    
    def _generate_triage_recommendations(self, 
                                       triage_category: str, 
                                       urgent_findings: Dict[str, Any], 
                                       parsed_symptoms: Dict[str, Any]) -> List[str]:
        """
        Generate triage recommendations based on category and findings.
        
        Args:
            triage_category: Determined triage category
            urgent_findings: Urgent findings from image analysis
            parsed_symptoms: Parsed symptom information
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Category-specific recommendations
        if triage_category == 'EMERGENCY':
            recommendations.extend([
                "🚨 EMERGENCY: Immediate medical attention required",
                "Contact emergency services or proceed to emergency department",
                "Patient requires immediate evaluation by healthcare provider"
            ])
        elif triage_category == 'URGENT':
            recommendations.extend([
                "⚠️ URGENT: Prompt medical evaluation needed",
                "Schedule appointment within 24-48 hours",
                "Monitor symptoms closely and seek care if worsening"
            ])
        elif triage_category == 'MODERATE':
            recommendations.extend([
                "📋 MODERATE: Standard medical evaluation appropriate",
                "Schedule appointment within 1-2 weeks",
                "Continue monitoring symptoms"
            ])
        elif triage_category == 'ROUTINE':
            recommendations.extend([
                "📅 ROUTINE: Standard follow-up recommended",
                "Schedule routine appointment as needed",
                "Continue current treatment plan"
            ])
        else:  # NORMAL
            recommendations.extend([
                "✅ NORMAL: No immediate concerns detected",
                "Continue with regular health monitoring",
                "Follow up with healthcare provider as scheduled"
            ])
        
        # Add specific findings recommendations
        if urgent_findings.get('urgent_pathologies'):
            recommendations.append("Specific findings detected - see detailed report")
        
        # Add symptom-based recommendations
        severity = parsed_symptoms.get('severity_classification', 'medium')
        if severity == 'emergency':
            recommendations.append("⚠️ Emergency symptoms detected - immediate evaluation required")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for triage assessment."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_triage_inputs(self, 
                              image_analysis: Dict[str, Any], 
                              symptoms: str) -> Dict[str, Any]:
        """
        Validate inputs for triage assessment.
        
        Args:
            image_analysis: Results from image analysis
            symptoms: Raw symptom text
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate image analysis
        if not image_analysis:
            validation_result['errors'].append("No image analysis results provided")
            validation_result['is_valid'] = False
        elif image_analysis.get('analysis_status') == 'failed':
            validation_result['errors'].append("Image analysis failed")
            validation_result['is_valid'] = False
        
        # Validate symptoms
        if not symptoms or symptoms.strip() == "":
            validation_result['warnings'].append("No symptoms provided - triage based on image only")
        
        return validation_result 