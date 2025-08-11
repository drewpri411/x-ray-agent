"""
Symptom Agent for Processing Patient Symptoms
Uses LLM to extract and categorize symptoms from patient text input.
"""

from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import logging
import re

logger = logging.getLogger(__name__)

class SymptomAgent:
    """Agent for parsing and categorizing patient symptoms."""
    
    def __init__(self, llm_model: str = "gemini-2.5-pro"):
        """
        Initialize the symptom agent.
        
        Args:
            llm_model: Gemini model to use for symptom parsing
        """
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompts for symptom parsing."""
        self.symptom_parser_prompt = ChatPromptTemplate.from_template("""
You are a medical assistant analyzing patient symptoms. Extract and categorize the symptoms from the patient's description.

Patient symptoms: {symptoms}

Please analyze the symptoms and return a structured response with the following information:

1. **Primary Symptoms**: List the main symptoms mentioned
2. **Severity Indicators**: Identify any severity indicators (mild, moderate, severe, etc.)
3. **Duration**: Extract any time-related information
4. **Risk Factors**: Identify any concerning symptoms that might indicate serious conditions
5. **Urgency Level**: Assess the urgency level (low, medium, high, emergency)

Focus on symptoms that are relevant to chest X-ray interpretation, such as:
- Respiratory symptoms (cough, shortness of breath, chest pain)
- Fever and systemic symptoms
- Cardiovascular symptoms
- Duration and progression of symptoms

Return your response in a clear, structured format.
""")

        self.severity_classifier_prompt = ChatPromptTemplate.from_template("""
Based on the following symptoms, classify the urgency level for chest X-ray evaluation:

Symptoms: {symptoms}

Classify as one of the following:
- EMERGENCY: Immediate attention required (severe chest pain, difficulty breathing, high fever with respiratory symptoms)
- HIGH: Urgent evaluation needed (moderate symptoms, concerning risk factors)
- MEDIUM: Standard evaluation appropriate (mild to moderate symptoms)
- LOW: Routine evaluation (minimal symptoms, stable condition)

Provide your classification and brief reasoning.
""")
    
    def parse_symptoms(self, symptoms_text: str) -> Dict[str, Any]:
        """
        Parse and categorize patient symptoms.
        
        Args:
            symptoms_text: Raw text describing patient symptoms
            
        Returns:
            Dictionary containing parsed symptom information
        """
        try:
            # Use LLM to parse symptoms
            response = self.llm.invoke(
                self.symptom_parser_prompt.format(symptoms=symptoms_text)
            )
            
            # Extract structured information from response
            parsed_symptoms = self._extract_symptom_info(response.content)
            
            # Add severity classification
            severity_response = self.llm.invoke(
                self.severity_classifier_prompt.format(symptoms=symptoms_text)
            )
            parsed_symptoms['severity_classification'] = self._extract_severity(severity_response.content)
            
            logger.info(f"Symptoms parsed successfully: {parsed_symptoms['severity_classification']}")
            return parsed_symptoms
            
        except Exception as e:
            logger.error(f"Failed to parse symptoms: {e}")
            # Fallback to basic parsing
            return self._fallback_symptom_parsing(symptoms_text)
    
    def _extract_symptom_info(self, llm_response: str) -> Dict[str, Any]:
        """Extract structured information from LLM response."""
        # Simple extraction - in production, you might use more sophisticated parsing
        symptoms_info = {
            'primary_symptoms': [],
            'severity_indicators': [],
            'duration': '',
            'risk_factors': [],
            'urgency_level': 'medium'
        }
        
        # Extract key information using regex patterns
        lines = llm_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'primary symptoms' in line.lower():
                current_section = 'primary_symptoms'
            elif 'severity indicators' in line.lower():
                current_section = 'severity_indicators'
            elif 'duration' in line.lower():
                current_section = 'duration'
            elif 'risk factors' in line.lower():
                current_section = 'risk_factors'
            elif 'urgency level' in line.lower():
                current_section = 'urgency_level'
            elif current_section and line.startswith('-') or line.startswith('•'):
                # Extract list items
                item = line[1:].strip()
                if current_section in ['primary_symptoms', 'severity_indicators', 'risk_factors']:
                    symptoms_info[current_section].append(item)
                elif current_section == 'duration':
                    symptoms_info['duration'] = item
                elif current_section == 'urgency_level':
                    symptoms_info['urgency_level'] = item.lower()
        
        return symptoms_info
    
    def _extract_severity(self, severity_response: str) -> str:
        """Extract severity classification from LLM response."""
        severity_levels = ['emergency', 'high', 'medium', 'low']
        
        for level in severity_levels:
            if level in severity_response.lower():
                return level
        
        return 'medium'  # Default
    
    def _fallback_symptom_parsing(self, symptoms_text: str) -> Dict[str, Any]:
        """Fallback symptom parsing using simple keyword matching."""
        symptoms_lower = symptoms_text.lower()
        
        # Define keyword patterns
        emergency_keywords = ['severe', 'emergency', 'critical', 'unable to breathe', 'chest pain']
        high_urgency_keywords = ['moderate', 'worsening', 'fever', 'cough', 'shortness of breath']
        respiratory_keywords = ['cough', 'breathing', 'chest', 'lung', 'respiratory']
        
        # Determine urgency level
        urgency_level = 'low'
        if any(keyword in symptoms_lower for keyword in emergency_keywords):
            urgency_level = 'emergency'
        elif any(keyword in symptoms_lower for keyword in high_urgency_keywords):
            urgency_level = 'high'
        elif any(keyword in symptoms_lower for keyword in respiratory_keywords):
            urgency_level = 'medium'
        
        return {
            'primary_symptoms': [symptoms_text],
            'severity_indicators': [],
            'duration': '',
            'risk_factors': [],
            'urgency_level': urgency_level,
            'severity_classification': urgency_level
        }
    
    def get_urgency_score(self, parsed_symptoms: Dict[str, Any]) -> float:
        """
        Convert urgency level to numerical score.
        
        Args:
            parsed_symptoms: Parsed symptom information
            
        Returns:
            Urgency score between 0 and 1
        """
        urgency_mapping = {
            'emergency': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        urgency_level = parsed_symptoms.get('severity_classification', 'medium')
        return urgency_mapping.get(urgency_level, 0.5) 