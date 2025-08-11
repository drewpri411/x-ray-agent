"""
Report Agent for Medical Report Generation
Uses LLM to generate comprehensive medical reports from analysis results.
"""

from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportAgent:
    """Agent for generating comprehensive medical reports."""
    
    def __init__(self, llm_model: str = "gemini-2.5-pro"):
        """
        Initialize the report agent.
        
        Args:
            llm_model: Gemini model to use for report generation
        """
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompts for report generation."""
        self.report_prompt = ChatPromptTemplate.from_template("""
You are a medical AI assistant generating a comprehensive chest X-ray analysis report. 
Based on the provided data, create a professional medical report.

**Patient Information:**
- Symptoms: {symptoms}
- Symptom Analysis: {parsed_symptoms}

**Image Analysis Results:**
- Key Findings: {key_findings}
- High Confidence Findings: {high_confidence_findings}
- Full Pathology Scores: {full_predictions}

**Triage Assessment:**
- Triage Category: {triage_category}
- Urgency Score: {urgency_score}
- Recommendations: {recommendations}

**Heatmap Information:**
- Generated heatmaps for: {heatmap_pathologies}

Please generate a comprehensive report that includes:

1. **Executive Summary**: Brief overview of findings and urgency level
2. **Clinical Findings**: Detailed analysis of detected pathologies
3. **Symptom Correlation**: How symptoms relate to imaging findings
4. **Triage Assessment**: Urgency evaluation and recommendations
5. **Clinical Recommendations**: Specific next steps for healthcare providers
6. **Limitations**: Important disclaimers about AI analysis

IMPORTANT: This is a prototype AI system for educational/demonstration purposes only. 
All findings should be reviewed by qualified healthcare professionals before clinical use.

Make the report professional, clear, and actionable for healthcare providers.
""")

        self.summary_prompt = ChatPromptTemplate.from_template("""
Create a concise executive summary for the chest X-ray analysis:

**Key Findings:**
{key_findings}

**Urgency Level:**
{triage_category} (Score: {urgency_score})

**Primary Recommendations:**
{recommendations}

Generate a 2-3 sentence summary suitable for quick clinical review.
""")
    
    def generate_report(self, 
                       image_analysis: Dict[str, Any],
                       triage_result: Dict[str, Any],
                       symptoms: str = "") -> Dict[str, Any]:
        """
        Generate a comprehensive medical report.
        
        Args:
            image_analysis: Results from image analysis
            triage_result: Results from triage assessment
            symptoms: Raw symptom text
            
        Returns:
            Dictionary containing the generated report
        """
        try:
            # Prepare data for report generation
            report_data = self._prepare_report_data(image_analysis, triage_result, symptoms)
            
            # Generate main report
            report_response = self.llm.invoke(
                self.report_prompt.format(**report_data)
            )
            
            # Generate executive summary
            summary_response = self.llm.invoke(
                self.summary_prompt.format(**report_data)
            )
            
            # Compile report
            report = {
                'executive_summary': summary_response.content.strip(),
                'full_report': report_response.content.strip(),
                'report_metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'model_used': 'gpt-4o-mini',
                    'report_version': '1.0'
                },
                'key_sections': self._extract_report_sections(report_response.content),
                'clinical_alerts': self._extract_clinical_alerts(triage_result)
            }
            
            logger.info("Medical report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                'executive_summary': f"Error generating report: {str(e)}",
                'full_report': f"Report generation failed due to: {str(e)}",
                'report_metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'error': str(e)
                },
                'key_sections': {},
                'clinical_alerts': []
            }
    
    def _prepare_report_data(self, 
                           image_analysis: Dict[str, Any],
                           triage_result: Dict[str, Any],
                           symptoms: str) -> Dict[str, Any]:
        """Prepare data for report generation."""
        # Extract key findings
        key_findings = image_analysis.get('key_findings', {})
        high_confidence = image_analysis.get('high_confidence_findings', {})
        full_predictions = image_analysis.get('classification_results', {}).get('full_predictions', {})
        
        # Format findings for report
        key_findings_text = self._format_findings(key_findings)
        high_confidence_text = self._format_findings(high_confidence)
        full_predictions_text = self._format_findings(full_predictions)
        
        # Get heatmap information
        heatmaps = image_analysis.get('heatmaps', {})
        heatmap_pathologies = list(heatmaps.keys()) if heatmaps else []
        
        # Get triage information
        triage_category = triage_result.get('triage_category', 'UNKNOWN')
        urgency_score = triage_result.get('urgency_score', 0.0)
        recommendations = triage_result.get('recommendations', [])
        
        # Get parsed symptoms
        parsed_symptoms = triage_result.get('parsed_symptoms', {})
        
        return {
            'symptoms': symptoms,
            'parsed_symptoms': self._format_parsed_symptoms(parsed_symptoms),
            'key_findings': key_findings_text,
            'high_confidence_findings': high_confidence_text,
            'full_predictions': full_predictions_text,
            'triage_category': triage_category,
            'urgency_score': f"{urgency_score:.3f}",
            'recommendations': '\n'.join(recommendations),
            'heatmap_pathologies': ', '.join(heatmap_pathologies) if heatmap_pathologies else 'None'
        }
    
    def _format_findings(self, findings: Dict[str, float]) -> str:
        """Format findings dictionary for report."""
        if not findings:
            return "No significant findings detected"
        
        formatted_items = []
        for pathology, score in findings.items():
            formatted_items.append(f"- {pathology}: {score:.3f}")
        
        return '\n'.join(formatted_items)
    
    def _format_parsed_symptoms(self, parsed_symptoms: Dict[str, Any]) -> str:
        """Format parsed symptoms for report."""
        if not parsed_symptoms:
            return "No symptoms provided"
        
        formatted = []
        
        # Primary symptoms
        primary = parsed_symptoms.get('primary_symptoms', [])
        if primary:
            formatted.append(f"Primary symptoms: {', '.join(primary)}")
        
        # Severity
        severity = parsed_symptoms.get('severity_classification', 'unknown')
        formatted.append(f"Severity classification: {severity}")
        
        # Duration
        duration = parsed_symptoms.get('duration', '')
        if duration:
            formatted.append(f"Duration: {duration}")
        
        # Risk factors
        risk_factors = parsed_symptoms.get('risk_factors', [])
        if risk_factors:
            formatted.append(f"Risk factors: {', '.join(risk_factors)}")
        
        return '\n'.join(formatted)
    
    def _extract_report_sections(self, report_text: str) -> Dict[str, str]:
        """Extract key sections from the generated report."""
        sections = {}
        
        # Simple section extraction based on headers
        lines = report_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if line.upper().startswith(('EXECUTIVE SUMMARY', 'CLINICAL FINDINGS', 
                                      'SYMPTOM CORRELATION', 'TRIAGE ASSESSMENT',
                                      'CLINICAL RECOMMENDATIONS', 'LIMITATIONS')):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_clinical_alerts(self, triage_result: Dict[str, Any]) -> List[str]:
        """Extract clinical alerts from triage results."""
        alerts = []
        
        # Add triage category alert
        triage_category = triage_result.get('triage_category', 'UNKNOWN')
        if triage_category in ['EMERGENCY', 'URGENT']:
            alerts.append(f"URGENT: {triage_category} triage level requires immediate attention")
        
        # Add urgent findings alerts
        urgent_findings = triage_result.get('urgent_findings', {})
        urgent_pathologies = urgent_findings.get('urgent_pathologies', [])
        
        for finding in urgent_pathologies:
            pathology = finding.get('pathology', '')
            score = finding.get('score', 0.0)
            alerts.append(f"ALERT: {pathology} detected with confidence {score:.3f}")
        
        return alerts
    
    def generate_follow_up_report(self, 
                                 original_report: Dict[str, Any],
                                 new_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a follow-up report based on new findings.
        
        Args:
            original_report: Original report data
            new_findings: New findings to incorporate
            
        Returns:
            Updated report
        """
        # This could be implemented to generate follow-up reports
        # For now, return a simple update
        return {
            'executive_summary': f"Follow-up analysis completed. {len(new_findings)} new findings identified.",
            'full_report': f"Follow-up report based on new findings: {new_findings}",
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'report_type': 'follow_up',
                'original_report_timestamp': original_report.get('report_metadata', {}).get('generation_timestamp', 'unknown')
            }
        } 