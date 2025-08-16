"""
Memory System for Clinical Learning and Pattern Recognition
"""

from typing import Dict, Any, List, Optional
import json
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ClinicalMemorySystem:
    """Memory system for storing and retrieving clinical cases and patterns."""
    
    def __init__(self, memory_file: str = "data/clinical_memory.json"):
        """Initialize the memory system."""
        self.memory_file = memory_file
        self.memory_data = self._load_memory()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._update_vectorizer()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory data from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memory file: {e}")
        
        # Initialize empty memory structure
        return {
            "cases": [],
            "patterns": {},
            "statistics": {
                "total_cases": 0,
                "diagnoses": {},
                "common_findings": {}
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_memory(self):
        """Save memory data to file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Convert numpy values to regular Python types for JSON serialization
            memory_data_serializable = self._convert_to_serializable(self.memory_data)
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data_serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to regular Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy type
            return float(obj)
        else:
            return obj
    
    def _update_vectorizer(self):
        """Update the TF-IDF vectorizer with current cases."""
        if not self.memory_data["cases"]:
            return
        
        # Create text representations of cases
        case_texts = []
        for case in self.memory_data["cases"]:
            text_parts = []
            
            # Add findings
            if "findings" in case:
                for finding, score in case["findings"].items():
                    if score > 0.3:  # Only include significant findings
                        text_parts.append(f"{finding}_{score:.1f}")
            
            # Add symptoms
            if "symptoms" in case:
                text_parts.extend(case["symptoms"])
            
            # Add diagnosis
            if "diagnosis" in case:
                text_parts.append(case["diagnosis"])
            
            case_texts.append(" ".join(text_parts))
        
        if case_texts:
            self.vectorizer.fit(case_texts)
    
    def add_case(self, case_data: Dict[str, Any]):
        """Add a new clinical case to memory."""
        case = {
            "id": f"case_{len(self.memory_data['cases']) + 1}",
            "timestamp": datetime.now().isoformat(),
            "findings": case_data.get("findings", {}),
            "symptoms": case_data.get("symptoms", []),
            "diagnosis": case_data.get("diagnosis", ""),
            "confidence": case_data.get("confidence", 0.0),
            "severity": case_data.get("severity", "moderate"),
            "outcome": case_data.get("outcome", ""),
            "learnings": case_data.get("learnings", [])
        }
        
        self.memory_data["cases"].append(case)
        self._update_statistics(case)
        self._update_vectorizer()
        self._save_memory()
        
        logger.info(f"Added case {case['id']} to memory")
        return case["id"]
    
    def _update_statistics(self, case: Dict[str, Any]):
        """Update memory statistics with new case."""
        stats = self.memory_data["statistics"]
        stats["total_cases"] += 1
        
        # Update diagnosis statistics
        diagnosis = case.get("diagnosis", "unknown")
        if diagnosis not in stats["diagnoses"]:
            stats["diagnoses"][diagnosis] = 0
        stats["diagnoses"][diagnosis] += 1
        
        # Update findings statistics
        findings = case.get("findings", {})
        for finding, score in findings.items():
            if score > 0.3:  # Only count significant findings
                if finding not in stats["common_findings"]:
                    stats["common_findings"][finding] = {"count": 0, "avg_score": 0.0}
                
                finding_stats = stats["common_findings"][finding]
                finding_stats["count"] += 1
                finding_stats["avg_score"] = (
                    (finding_stats["avg_score"] * (finding_stats["count"] - 1) + score) / 
                    finding_stats["count"]
                )
    
    def search_similar_cases(self, query_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar cases based on findings and symptoms."""
        if not self.memory_data["cases"]:
            return []
        
        # Create query text
        query_parts = []
        
        # Add findings
        findings = query_data.get("findings", {})
        for finding, score in findings.items():
            if score > 0.3:
                query_parts.append(f"{finding}_{score:.1f}")
        
        # Add symptoms
        symptoms = query_data.get("symptoms", [])
        query_parts.extend(symptoms)
        
        query_text = " ".join(query_parts)
        
        if not query_text.strip():
            return []
        
        try:
            # Vectorize query and cases
            query_vector = self.vectorizer.transform([query_text])
            case_texts = []
            
            for case in self.memory_data["cases"]:
                text_parts = []
                for finding, score in case.get("findings", {}).items():
                    if score > 0.3:
                        text_parts.append(f"{finding}_{score:.1f}")
                text_parts.extend(case.get("symptoms", []))
                if case.get("diagnosis"):
                    text_parts.append(case["diagnosis"])
                case_texts.append(" ".join(text_parts))
            
            if not case_texts:
                return []
            
            case_vectors = self.vectorizer.transform(case_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, case_vectors).flatten()
            
            # Get top similar cases
            similar_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_cases = []
            for idx in similar_indices:
                if similarities[idx] > 0.1:  # Only include reasonably similar cases
                    case = self.memory_data["cases"][idx].copy()
                    case["similarity"] = float(similarities[idx])
                    similar_cases.append(case)
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_patterns(self, diagnosis: str = None) -> Dict[str, Any]:
        """Get clinical patterns for a specific diagnosis or all patterns."""
        if diagnosis:
            # Get patterns for specific diagnosis
            diagnosis_cases = [
                case for case in self.memory_data["cases"] 
                if case.get("diagnosis", "").lower() == diagnosis.lower()
            ]
            
            if not diagnosis_cases:
                return {"error": f"No cases found for diagnosis: {diagnosis}"}
            
            return self._analyze_patterns(diagnosis_cases)
        else:
            # Get all patterns
            return self._analyze_patterns(self.memory_data["cases"])
    
    def _analyze_patterns(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in a set of cases."""
        if not cases:
            return {"error": "No cases to analyze"}
        
        patterns = {
            "total_cases": len(cases),
            "diagnoses": {},
            "common_findings": {},
            "symptom_patterns": {},
            "severity_distribution": {},
            "confidence_distribution": {}
        }
        
        # Analyze diagnoses
        for case in cases:
            diagnosis = case.get("diagnosis", "unknown")
            if diagnosis not in patterns["diagnoses"]:
                patterns["diagnoses"][diagnosis] = 0
            patterns["diagnoses"][diagnosis] += 1
        
        # Analyze findings
        for case in cases:
            findings = case.get("findings", {})
            for finding, score in findings.items():
                if score > 0.3:
                    if finding not in patterns["common_findings"]:
                        patterns["common_findings"][finding] = {"count": 0, "scores": []}
                    patterns["common_findings"][finding]["count"] += 1
                    patterns["common_findings"][finding]["scores"].append(score)
        
        # Calculate average scores
        for finding, data in patterns["common_findings"].items():
            data["avg_score"] = np.mean(data["scores"])
            data["std_score"] = np.std(data["scores"])
        
        # Analyze symptoms
        for case in cases:
            symptoms = case.get("symptoms", [])
            for symptom in symptoms:
                if symptom not in patterns["symptom_patterns"]:
                    patterns["symptom_patterns"][symptom] = 0
                patterns["symptom_patterns"][symptom] += 1
        
        # Analyze severity and confidence
        for case in cases:
            severity = case.get("severity", "unknown")
            if severity not in patterns["severity_distribution"]:
                patterns["severity_distribution"][severity] = 0
            patterns["severity_distribution"][severity] += 1
            
            confidence = case.get("confidence", 0.0)
            confidence_bucket = f"{int(confidence * 10) * 10}-{(int(confidence * 10) + 1) * 10}%"
            if confidence_bucket not in patterns["confidence_distribution"]:
                patterns["confidence_distribution"][confidence_bucket] = 0
            patterns["confidence_distribution"][confidence_bucket] += 1
        
        return patterns
    
    def get_recommendations(self, current_case: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on similar cases and patterns."""
        similar_cases = self.search_similar_cases(current_case, top_k=3)
        
        recommendations = {
            "similar_cases": similar_cases,
            "diagnostic_suggestions": [],
            "treatment_patterns": [],
            "risk_factors": [],
            "confidence_insights": []
        }
        
        if similar_cases:
            # Analyze diagnostic patterns
            diagnoses = [case.get("diagnosis") for case in similar_cases if case.get("diagnosis")]
            if diagnoses:
                most_common_diagnosis = max(set(diagnoses), key=diagnoses.count)
                recommendations["diagnostic_suggestions"].append({
                    "diagnosis": most_common_diagnosis,
                    "frequency": diagnoses.count(most_common_diagnosis),
                    "total_similar_cases": len(similar_cases)
                })
            
            # Analyze severity patterns
            severities = [case.get("severity") for case in similar_cases if case.get("severity")]
            if severities:
                most_common_severity = max(set(severities), key=severities.count)
                recommendations["severity_patterns"].append({
                    "severity": most_common_severity,
                    "frequency": severities.count(most_common_severity)
                })
            
            # Analyze confidence patterns
            confidences = [case.get("confidence", 0.0) for case in similar_cases]
            if confidences:
                avg_confidence = np.mean(confidences)
                recommendations["confidence_insights"].append({
                    "average_confidence": avg_confidence,
                    "confidence_range": f"{min(confidences):.2f}-{max(confidences):.2f}"
                })
        
        return recommendations
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        return {
            "total_cases": self.memory_data["statistics"]["total_cases"],
            "diagnoses": self.memory_data["statistics"]["diagnoses"],
            "common_findings": self.memory_data["statistics"]["common_findings"],
            "last_updated": self.memory_data["last_updated"]
        }
    
    def clear_memory(self):
        """Clear all memory data."""
        self.memory_data = {
            "cases": [],
            "patterns": {},
            "statistics": {
                "total_cases": 0,
                "diagnoses": {},
                "common_findings": {}
            },
            "last_updated": datetime.now().isoformat()
        }
        self._save_memory()
        logger.info("Memory cleared") 