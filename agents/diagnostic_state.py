"""
Diagnostic State Management for ReAct-Style Diagnostic Reasoning
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiagnosticState:
    """State management for ReAct-style diagnostic reasoning workflow."""
    
    # Core input data
    image_path: str
    symptoms: str = ""
    patient_info: Dict[str, Any] = field(default_factory=dict)
    
    # ReAct loop state
    current_step: str = "observe"  # "observe", "think", "act", "reflect", "conclude"
    iteration_count: int = 0
    max_iterations: int = 5
    should_continue: bool = True
    
    # Evidence and observations
    image_findings: Dict[str, Any] = field(default_factory=dict)
    clinical_observations: List[str] = field(default_factory=list)
    parsed_symptoms: Dict[str, Any] = field(default_factory=dict)
    
    # Diagnostic reasoning
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence_for: Dict[str, List[str]] = field(default_factory=dict)
    evidence_against: Dict[str, List[str]] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Tool usage tracking
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    selected_tools: List[str] = field(default_factory=list)
    
    # Memory and context
    clinical_memory: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Final outputs (preserve existing functionality)
    primary_diagnosis: Optional[str] = None
    differential_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    triage_category: Optional[str] = None
    urgency_score: float = 0.0
    final_recommendations: List[str] = field(default_factory=list)
    
    # Legacy compatibility (for existing workflow)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    triage_result: Dict[str, Any] = field(default_factory=dict)
    report_result: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    assessment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    workflow_status: str = "initialized"
    errors: List[str] = field(default_factory=list)
    
    def add_reasoning_step(self, step: str, reasoning: Dict[str, Any]):
        """Add a reasoning step to the chain."""
        self.reasoning_chain.append({
            'step': step,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat(),
            'reasoning': reasoning
        })
    
    def add_tool_usage(self, tool_name: str, tool_input: Dict[str, Any], tool_output: Dict[str, Any]):
        """Track tool usage."""
        self.tools_used.append({
            'tool_name': tool_name,
            'input': tool_input,
            'output': tool_output,
            'timestamp': datetime.now().isoformat(),
            'iteration': self.iteration_count
        })
        self.tool_results[tool_name] = tool_output
    
    def update_hypothesis(self, diagnosis: str, confidence: float, evidence: List[str], 
                         contradicting_evidence: List[str] = None):
        """Update or add a diagnostic hypothesis."""
        # Find existing hypothesis
        existing_idx = None
        for i, hyp in enumerate(self.hypotheses):
            if hyp['diagnosis'] == diagnosis:
                existing_idx = i
                break
        
        hypothesis_data = {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'evidence': evidence,
            'contradicting_evidence': contradicting_evidence or [],
            'iteration': self.iteration_count
        }
        
        if existing_idx is not None:
            self.hypotheses[existing_idx] = hypothesis_data
        else:
            self.hypotheses.append(hypothesis_data)
        
        self.confidence_scores[diagnosis] = confidence
        self.evidence_for[diagnosis] = evidence
        if contradicting_evidence:
            self.evidence_against[diagnosis] = contradicting_evidence
    
    def get_top_hypothesis(self) -> Optional[Dict[str, Any]]:
        """Get the hypothesis with highest confidence."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda x: x['confidence'])
    
    def should_stop_reasoning(self) -> bool:
        """Determine if reasoning should stop."""
        if self.iteration_count >= self.max_iterations:
            return True
        
        # Stop if we have any hypothesis with high confidence
        top_hypothesis = self.get_top_hypothesis()
        if top_hypothesis and top_hypothesis['confidence'] > 0.8:
            return True
        
        # Stop if we've made progress but confidence is not improving
        if self.iteration_count >= 2:
            return True
        
        return False
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            'workflow_status': self.workflow_status,
            'image_analysis': {
                'image_path': self.image_path,
                'key_findings': self.image_findings.get('key_findings', {}),
                'analysis_status': 'completed' if self.image_findings else 'pending'
            },
            'triage_result': {
                'triage_category': self.triage_category,
                'urgency_score': self.urgency_score,
                'recommendations': self.final_recommendations
            },
            'diagnostic_result': {
                'primary_diagnosis': self.primary_diagnosis,
                'differential_diagnoses': self.differential_diagnoses,
                'confidence_scores': self.confidence_scores
            },
            'reasoning_chain': self.reasoning_chain,
            'tools_used': self.tools_used,
            'errors': self.errors
        } 