"""
Diagnostic Tools Registry for ReAct-Style Diagnostic Reasoning
"""

from typing import Dict, Any, List, Optional
from langchain.tools import tool
import logging
import json
import re

logger = logging.getLogger(__name__)

class DiagnosticTools:
    """Registry of diagnostic tools for clinical reasoning."""
    
    def __init__(self):
        """Initialize diagnostic tools."""
        self.medical_knowledge = self._load_medical_knowledge()
        self.clinical_guidelines = self._load_clinical_guidelines()
        self.decision_rules = self._load_decision_rules()
    
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base."""
        return {
            "pneumonia": {
                "definition": "Infection of the lung parenchyma",
                "common_causes": ["bacterial", "viral", "fungal"],
                "risk_factors": ["age > 65", "smoking", "COPD", "immunosuppression"],
                "typical_findings": ["consolidation", "infiltrates", "effusion"],
                "differential_diagnosis": ["heart_failure", "pulmonary_embolism", "tuberculosis"]
            },
            "heart_failure": {
                "definition": "Inability of the heart to pump blood effectively",
                "types": ["systolic", "diastolic"],
                "risk_factors": ["hypertension", "coronary_artery_disease", "diabetes"],
                "typical_findings": ["cardiomegaly", "pulmonary_edema", "pleural_effusion"],
                "differential_diagnosis": ["pneumonia", "COPD_exacerbation", "pulmonary_embolism"]
            },
            "pleural_effusion": {
                "definition": "Accumulation of fluid in the pleural space",
                "causes": ["heart_failure", "pneumonia", "malignancy", "tuberculosis"],
                "types": ["transudative", "exudative"],
                "typical_findings": ["blunting_costophrenic_angles", "meniscus_sign"],
                "differential_diagnosis": ["pneumonia", "heart_failure", "malignancy"]
            },
            "atelectasis": {
                "definition": "Collapse of lung tissue affecting gas exchange",
                "causes": ["mucus_plug", "tumor", "foreign_body", "postoperative"],
                "types": ["subsegmental", "segmental", "lobar"],
                "typical_findings": ["volume_loss", "displacement_fissures", "elevated_hemidiaphragm", "mediastinal_shift"],
                "differential_diagnosis": ["pneumonia", "pneumothorax", "pleural_effusion"]
            }
        }
    
    def _load_clinical_guidelines(self) -> Dict[str, Any]:
        """Load clinical practice guidelines."""
        return {
            "pneumonia": {
                "severity_assessment": "CURB-65 score",
                "treatment": {
                    "outpatient": ["macrolide", "doxycycline", "beta_lactam"],
                    "inpatient": ["beta_lactam + macrolide", "respiratory_fluoroquinolone"],
                    "icu": ["beta_lactam + macrolide + fluoroquinolone"]
                },
                "imaging": "Chest X-ray for diagnosis and follow-up"
            },
            "heart_failure": {
                "severity_assessment": "NYHA classification",
                "treatment": {
                    "acute": ["diuretics", "vasodilators", "inotropes"],
                    "chronic": ["ACE_inhibitors", "beta_blockers", "aldosterone_antagonists"]
                },
                "imaging": "Echocardiography for ejection fraction"
            },
            "atelectasis": {
                "severity_assessment": "Extent of lung involvement",
                "treatment": {
                    "mild": ["deep_breathing_exercises", "incentive_spirometry"],
                    "moderate": ["bronchoscopy", "chest_physiotherapy"],
                    "severe": ["mechanical_ventilation", "surgical_intervention"]
                },
                "imaging": "Chest X-ray for diagnosis and follow-up"
            }
        }
    
    def _load_decision_rules(self) -> Dict[str, Any]:
        """Load clinical decision rules."""
        return {
            "curb65": {
                "components": ["confusion", "uremia", "respiratory_rate", "blood_pressure", "age"],
                "scoring": {
                    "0-1": "low_risk",
                    "2": "moderate_risk", 
                    "3-5": "high_risk"
                }
            },
            "nyha": {
                "class_i": "No limitation of physical activity",
                "class_ii": "Slight limitation of physical activity",
                "class_iii": "Marked limitation of physical activity",
                "class_iv": "Unable to carry on any physical activity"
            }
        }

@tool
def differential_diagnosis_tool(observations: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Look up differential diagnoses for given clinical observations.
    
    Args:
        observations: List of clinical observations and findings
        
    Returns:
        Dictionary containing differential diagnoses with confidence levels
    """
    tools = DiagnosticTools()
    
    # Extract observations from dict
    observations_list = observations.get("observations", [])
    observation_text = " ".join(observations_list).lower()
    
    differentials = []
    
    # Check each condition in medical knowledge
    for condition, info in tools.medical_knowledge.items():
        relevance_score = 0
        supporting_evidence = []
        
        # Check if condition's typical findings are mentioned
        for finding in info.get("typical_findings", []):
            if finding.lower() in observation_text:
                relevance_score += 0.3
                supporting_evidence.append(f"Finding: {finding}")
        
        # Check if condition is mentioned directly
        if condition.lower() in observation_text:
            relevance_score += 0.5
            supporting_evidence.append(f"Direct mention: {condition}")
        
        # Check differential diagnosis relationships
        for diff_condition in info.get("differential_diagnosis", []):
            if diff_condition.lower() in observation_text:
                relevance_score += 0.2
                supporting_evidence.append(f"Related to: {diff_condition}")
        
        if relevance_score > 0:
            differentials.append({
                "diagnosis": condition,
                "confidence": min(relevance_score, 1.0),
                "supporting_evidence": supporting_evidence,
                "definition": info.get("definition", ""),
                "risk_factors": info.get("risk_factors", [])
            })
    
    # Sort by confidence
    differentials.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "differential_diagnoses": differentials[:5],  # Top 5
        "total_conditions_checked": len(tools.medical_knowledge),
        "observation_text": observation_text
    }

@tool
def clinical_guidelines_tool(diagnosis: str, severity: str = "moderate") -> Dict[str, Any]:
    """
    Access clinical practice guidelines for a specific diagnosis.
    
    Args:
        diagnosis: The primary diagnosis
        severity: Severity level (mild, moderate, severe)
        
    Returns:
        Dictionary containing treatment guidelines and recommendations
    """
    tools = DiagnosticTools()
    diagnosis_lower = diagnosis.lower()
    
    if diagnosis_lower not in tools.clinical_guidelines:
        return {
            "error": f"No guidelines found for diagnosis: {diagnosis}",
            "available_guidelines": list(tools.clinical_guidelines.keys())
        }
    
    guidelines = tools.clinical_guidelines[diagnosis_lower]
    
    # Select treatment based on severity
    treatment_options = guidelines.get("treatment", {})
    recommended_treatment = None
    
    if severity == "mild":
        recommended_treatment = treatment_options.get("outpatient", [])
    elif severity == "moderate":
        recommended_treatment = treatment_options.get("inpatient", treatment_options.get("outpatient", []))
    elif severity == "severe":
        recommended_treatment = treatment_options.get("icu", treatment_options.get("inpatient", []))
    
    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "severity_assessment": guidelines.get("severity_assessment", ""),
        "recommended_treatment": recommended_treatment,
        "imaging_recommendations": guidelines.get("imaging", ""),
        "all_treatment_options": treatment_options
    }

@tool
def clinical_decision_rules_tool(rule_name: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply clinical decision rules (e.g., CURB-65, NYHA) to patient data.
    
    Args:
        rule_name: Name of the decision rule (e.g., "curb65", "nyha")
        patient_data: Patient clinical data
        
    Returns:
        Dictionary containing rule application results
    """
    tools = DiagnosticTools()
    rule_name_lower = rule_name.lower()
    
    if rule_name_lower not in tools.decision_rules:
        return {
            "error": f"Decision rule not found: {rule_name}",
            "available_rules": list(tools.decision_rules.keys())
        }
    
    rule = tools.decision_rules[rule_name_lower]
    
    if rule_name_lower == "curb65":
        return _apply_curb65_rule(patient_data, rule)
    elif rule_name_lower == "nyha":
        return _apply_nyha_rule(patient_data, rule)
    
    return {
        "error": f"Rule application not implemented for: {rule_name}",
        "rule_info": rule
    }

def _apply_curb65_rule(patient_data: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CURB-65 scoring rule."""
    score = 0
    components = {}
    
    # Confusion
    if patient_data.get("confusion", False):
        score += 1
        components["confusion"] = True
    
    # Urea > 7 mmol/L
    if patient_data.get("urea", 0) > 7:
        score += 1
        components["uremia"] = True
    
    # Respiratory rate >= 30/min
    if patient_data.get("respiratory_rate", 0) >= 30:
        score += 1
        components["respiratory_rate"] = True
    
    # Blood pressure (systolic < 90 or diastolic <= 60)
    if (patient_data.get("systolic_bp", 0) < 90 or 
        patient_data.get("diastolic_bp", 0) <= 60):
        score += 1
        components["blood_pressure"] = True
    
    # Age >= 65
    if patient_data.get("age", 0) >= 65:
        score += 1
        components["age"] = True
    
    # Determine risk category
    risk_category = "low_risk"
    if score >= 3:
        risk_category = "high_risk"
    elif score == 2:
        risk_category = "moderate_risk"
    
    return {
        "rule": "CURB-65",
        "score": score,
        "risk_category": risk_category,
        "components": components,
        "interpretation": rule["scoring"].get(str(score), "unknown")
    }

def _apply_nyha_rule(patient_data: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """Apply NYHA classification rule."""
    # This is a simplified implementation
    # In practice, NYHA classification requires detailed clinical assessment
    
    symptoms = patient_data.get("symptoms", "").lower()
    functional_capacity = patient_data.get("functional_capacity", "")
    
    if "no limitation" in functional_capacity or "asymptomatic" in symptoms:
        nyha_class = "I"
    elif "slight limitation" in functional_capacity or "mild symptoms" in symptoms:
        nyha_class = "II"
    elif "marked limitation" in functional_capacity or "moderate symptoms" in symptoms:
        nyha_class = "III"
    elif "unable" in functional_capacity or "severe symptoms" in symptoms:
        nyha_class = "IV"
    else:
        nyha_class = "Unknown"
    
    return {
        "rule": "NYHA",
        "class": nyha_class,
        "description": rule.get(f"class_{nyha_class.lower()}", "Unknown"),
        "functional_capacity": functional_capacity,
        "symptoms": symptoms
    }

@tool
def medical_knowledge_lookup_tool(condition: str) -> Dict[str, Any]:
    """
    Look up detailed medical knowledge about a specific condition.
    
    Args:
        condition: The medical condition to look up
        
    Returns:
        Dictionary containing detailed medical information
    """
    tools = DiagnosticTools()
    condition_lower = condition.lower()
    
    if condition_lower not in tools.medical_knowledge:
        return {
            "error": f"Condition not found: {condition}",
            "available_conditions": list(tools.medical_knowledge.keys())
        }
    
    return {
        "condition": condition,
        "information": tools.medical_knowledge[condition_lower]
    }

@tool
def severity_assessment_tool(image_findings: Dict[str, Any], symptoms: str) -> Dict[str, Any]:
    """
    Assess severity of findings based on image and symptoms.
    
    Args:
        image_findings: Results from image analysis
        symptoms: Patient symptoms
        
    Returns:
        Dictionary containing severity assessment
    """
    severity_score = 0.0
    severity_factors = []
    
    # Analyze image findings
    key_findings = image_findings.get('key_findings', {})
    
    # High-risk pathologies
    high_risk_pathologies = ['pneumonia', 'effusion', 'cardiomegaly', 'edema']
    for pathology in high_risk_pathologies:
        if pathology in key_findings:
            score = key_findings[pathology]
            if score > 0.8:
                severity_score += 0.4
                severity_factors.append(f"High {pathology} (score: {score:.2f})")
            elif score > 0.6:
                severity_score += 0.2
                severity_factors.append(f"Moderate {pathology} (score: {score:.2f})")
    
    # Analyze symptoms
    symptoms_lower = symptoms.lower()
    emergency_symptoms = ['severe', 'acute', 'emergency', 'unable to breathe']
    for symptom in emergency_symptoms:
        if symptom in symptoms_lower:
            severity_score += 0.3
            severity_factors.append(f"Emergency symptom: {symptom}")
    
    # Determine severity category
    if severity_score >= 0.8:
        severity_category = "severe"
    elif severity_score >= 0.5:
        severity_category = "moderate"
    elif severity_score >= 0.2:
        severity_category = "mild"
    else:
        severity_category = "minimal"
    
    return {
        "severity_score": min(severity_score, 1.0),
        "severity_category": severity_category,
        "severity_factors": severity_factors,
        "image_findings": key_findings,
        "symptoms_analyzed": symptoms
    } 