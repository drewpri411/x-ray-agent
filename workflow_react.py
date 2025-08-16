"""
ReAct-Style Diagnostic Reasoning Workflow using LangGraph
"""

from typing import Dict, Any
import logging
from langgraph.graph import StateGraph, START, END
from agents.diagnostic_state import DiagnosticState
from agents.react_nodes import ReActNodes

logger = logging.getLogger(__name__)

def create_react_diagnostic_workflow():
    """Create the ReAct-style diagnostic reasoning workflow."""
    
    # Initialize ReAct nodes
    react_nodes = ReActNodes()
    
    # Create state graph
    workflow = StateGraph(DiagnosticState)
    
    # Add nodes
    workflow.add_node("observe", react_nodes.observe_node)
    workflow.add_node("retrieve_memory", react_nodes.retrieve_memory_node)
    workflow.add_node("think", react_nodes.think_node)
    workflow.add_node("select_tools", react_nodes.select_tools_node)
    workflow.add_node("execute_tools", react_nodes.execute_tools_node)
    workflow.add_node("reflect", react_nodes.reflect_node)
    workflow.add_node("update_memory", react_nodes.update_memory_node)
    workflow.add_node("conclude", react_nodes.conclude_node)
    
    # Define conditional edges
    def should_continue_condition(state: DiagnosticState) -> str:
        """Determine if the workflow should continue or conclude."""
        if state.should_continue and state.iteration_count < state.max_iterations:
            return "continue"
        else:
            return "conclude"
    
    def needs_more_info_condition(state: DiagnosticState) -> str:
        """Determine if more information is needed."""
        # Always use tools to validate and enhance hypotheses
        return "select_tools"  # Always select and execute tools for better reasoning
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "observe",
        should_continue_condition,
        {
            "continue": "retrieve_memory",
            "conclude": "conclude"
        }
    )
    
    workflow.add_conditional_edges(
        "think",
        needs_more_info_condition,
        {
            "select_tools": "select_tools",
            "reflect": "reflect"
        }
    )
    
    # Add entry point
    workflow.add_edge(START, "observe")
    
    # Add linear edges
    workflow.add_edge("retrieve_memory", "think")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "reflect")
    workflow.add_edge("reflect", "update_memory")
    workflow.add_edge("update_memory", "observe")
    workflow.add_edge("conclude", END)
    
    return workflow.compile()

class ReActDiagnosticWorkflowRunner:
    """Runner for the ReAct-style diagnostic reasoning workflow."""
    
    def __init__(self):
        """Initialize the workflow runner."""
        self.workflow = create_react_diagnostic_workflow()
        logger.info("ReAct diagnostic workflow initialized")
    
    def run_diagnostic_analysis(self, 
                               image_path: str, 
                               symptoms: str = "", 
                               patient_info: Dict[str, Any] = None,
                               generate_heatmaps: bool = False) -> Dict[str, Any]:
        """
        Run the complete ReAct-style diagnostic analysis.
        
        Args:
            image_path: Path to the X-ray image
            symptoms: Patient symptoms
            patient_info: Additional patient information
            generate_heatmaps: Whether to generate Grad-CAM heatmaps
            
        Returns:
            Dictionary containing complete diagnostic analysis results
        """
        try:
            logger.info(f"Starting ReAct diagnostic analysis for: {image_path}")
            
            # Initialize state
            initial_state = DiagnosticState(
                image_path=image_path,
                symptoms=symptoms,
                patient_info=patient_info or {}
            )
            
            # Run workflow
            try:
                final_state = self.workflow.invoke(initial_state)
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                # Return initial state with error
                initial_state.workflow_status = "failed"
                initial_state.errors.append(str(e))
                final_state = initial_state
            
            # Handle case where LangGraph returns a dictionary instead of state object
            if isinstance(final_state, dict):
                logger.info("Workflow returned dictionary, extracting state")
                # LangGraph sometimes returns a dictionary with the state
                if 'state' in final_state:
                    final_state = final_state['state']
                else:
                    # If it's a dictionary but not a state object, create a result directly
                    logger.warning("Workflow returned unexpected dictionary format")
                    return {
                        "workflow_status": "completed",
                        "react_workflow": {
                            "iteration_count": final_state.get('iteration_count', 0),
                            "reasoning_chain": final_state.get('reasoning_chain', []),
                            "tools_used": final_state.get('tools_used', []),
                            "hypotheses": final_state.get('hypotheses', []),
                            "clinical_memory": final_state.get('clinical_memory', []),
                            "similar_cases": final_state.get('similar_cases', [])
                        },
                        "diagnostic_insights": {
                            "primary_diagnosis": final_state.get('primary_diagnosis'),
                            "differential_diagnoses": final_state.get('differential_diagnoses', []),
                            "confidence_scores": final_state.get('confidence_scores', {}),
                            "evidence_for": final_state.get('evidence_for', {}),
                            "evidence_against": final_state.get('evidence_against', {})
                        },
                        "clinical_recommendations": {
                            "triage_category": final_state.get('triage_category'),
                            "urgency_score": final_state.get('urgency_score', 0.0),
                            "recommendations": final_state.get('final_recommendations', [])
                        }
                    }
            
            # Ensure final_state is a DiagnosticState object
            if not hasattr(final_state, 'to_legacy_format'):
                logger.error("Final state is not a DiagnosticState object")
                return {
                    "workflow_status": "failed",
                    "error": "Invalid state object returned from workflow",
                    "react_workflow": {
                        "iteration_count": 0,
                        "reasoning_chain": [],
                        "tools_used": [],
                        "hypotheses": []
                    }
                }
            
            # Convert to legacy format for backward compatibility
            result = final_state.to_legacy_format()
            
            # Add ReAct-specific information
            result.update({
                "react_workflow": {
                    "iteration_count": final_state.iteration_count,
                    "reasoning_chain": final_state.reasoning_chain,
                    "tools_used": final_state.tools_used,
                    "hypotheses": final_state.hypotheses,
                    "clinical_memory": final_state.clinical_memory,
                    "similar_cases": final_state.similar_cases
                },
                "diagnostic_insights": {
                    "primary_diagnosis": final_state.primary_diagnosis,
                    "differential_diagnoses": final_state.differential_diagnoses,
                    "confidence_scores": final_state.confidence_scores,
                    "evidence_for": final_state.evidence_for,
                    "evidence_against": final_state.evidence_against
                },
                "clinical_recommendations": {
                    "triage_category": final_state.triage_category,
                    "urgency_score": final_state.urgency_score,
                    "recommendations": final_state.final_recommendations
                }
            })
            
            logger.info(f"ReAct diagnostic analysis completed: {final_state.primary_diagnosis}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run ReAct diagnostic analysis: {e}")
            return {
                "workflow_status": "failed",
                "error": str(e),
                "react_workflow": {
                    "iteration_count": 0,
                    "reasoning_chain": [],
                    "tools_used": [],
                    "hypotheses": []
                }
            }

    def _state_snapshot(self, state: DiagnosticState) -> Dict[str, Any]:
        """Create a lightweight, serializable snapshot of the current state for UI streaming."""
        try:
            # Safely convert numpy types to Python floats
            findings = {}
            for k, v in (state.image_findings.get("key_findings", {}) or {}).items():
                try:
                    findings[k] = float(v) if hasattr(v, 'item') else (float(v) if isinstance(v, (int, float)) else v)
                except Exception:
                    findings[k] = v

            return {
                "iteration": state.iteration_count,
                "should_continue": state.should_continue,
                "image_findings": findings,
                "clinical_observations": list(state.clinical_observations) if state.clinical_observations else [],
                "selected_tools": list(state.selected_tools) if hasattr(state, 'selected_tools') else [],
                "tools_used": list(state.tools_used) if hasattr(state, 'tools_used') else [],
                "hypotheses": state.hypotheses[:5] if state.hypotheses else [],
                "similar_cases": state.similar_cases[:3] if state.similar_cases else [],
                "errors": list(state.errors) if state.errors else [],
            }
        except Exception:
            return {}

    def _final_result_from_state(self, state: DiagnosticState) -> Dict[str, Any]:
        """Build final result dict from state (mirrors run_diagnostic_analysis)."""
        result = state.to_legacy_format()
        result.update({
            "react_workflow": {
                "iteration_count": state.iteration_count,
                "reasoning_chain": state.reasoning_chain,
                "tools_used": state.tools_used,
                "hypotheses": state.hypotheses,
                "clinical_memory": state.clinical_memory,
                "similar_cases": state.similar_cases
            },
            "diagnostic_insights": {
                "primary_diagnosis": state.primary_diagnosis,
                "differential_diagnoses": state.differential_diagnoses,
                "confidence_scores": state.confidence_scores,
                "evidence_for": state.evidence_for,
                "evidence_against": state.evidence_against
            },
            "clinical_recommendations": {
                "triage_category": state.triage_category,
                "urgency_score": state.urgency_score,
                "recommendations": state.final_recommendations
            }
        })
        return result

    def run_diagnostic_analysis_stream(self,
                                       image_path: str,
                                       symptoms: str = "",
                                       patient_info: Dict[str, Any] = None,
                                       generate_heatmaps: bool = False):
        """Generator that yields stage-by-stage progress events for live UI updates.

        Yields dicts like: {"stage": str, "message": str, "snapshot": {...}}
        The last yield will include {"stage": "conclude", "final_result": {...}}
        """
        react_nodes = ReActNodes()

        # Conditions consistent with the compiled graph
        def should_continue_condition(state: DiagnosticState) -> str:
            if state.should_continue and state.iteration_count < state.max_iterations:
                return "continue"
            return "conclude"

        # Always route to select_tools after think (as per current design)
        def needs_more_info_condition(_: DiagnosticState) -> str:
            return "select_tools"

        # Initialize state
        state = DiagnosticState(
            image_path=image_path,
            symptoms=symptoms,
            patient_info=patient_info or {}
        )

        # Precompute image analysis (with optional heatmaps) before first observe
        try:
            pre_image_results = react_nodes.image_agent.analyze_image(
                image_path,
                generate_heatmaps=generate_heatmaps
            )
            state.image_findings = pre_image_results
        except Exception:
            # Fallback to within-node analysis
            pass

        # START -> observe
        state = react_nodes.observe_node(state)
        first_event: Dict[str, Any] = {
            "stage": "observe",
            "message": "Analyzing image and extracting observations",
            "snapshot": self._state_snapshot(state),
            "image_path": image_path,
        }
        # Attach heatmaps if available
        if isinstance(state.image_findings, dict) and 'heatmaps' in state.image_findings:
            first_event["heatmaps"] = state.image_findings.get('heatmaps', {})
        yield first_event

        # Conditional edge from observe
        branch = should_continue_condition(state)
        if branch == "conclude":
            state = react_nodes.conclude_node(state)
            final = self._final_result_from_state(state)
            yield {
                "stage": "conclude",
                "message": "Finalizing diagnosis and recommendations",
                "snapshot": self._state_snapshot(state),
                "final_result": final
            }
            return

        # Main loop mirrors retrieve_memory -> think -> select_tools -> execute_tools -> reflect -> update_memory -> observe
        while True:
            state = react_nodes.retrieve_memory_node(state)
            yield {
                "stage": "retrieve_memory",
                "message": "Retrieving similar cases from clinical memory",
                "snapshot": self._state_snapshot(state)
            }

            state = react_nodes.think_node(state)
            yield {
                "stage": "think",
                "message": "Generating diagnostic hypotheses",
                "snapshot": self._state_snapshot(state)
            }

            # Decide next from think (current logic always selects tools)
            _ = needs_more_info_condition(state)

            state = react_nodes.select_tools_node(state)
            yield {
                "stage": "select_tools",
                "message": "Selecting diagnostic tools",
                "snapshot": self._state_snapshot(state)
            }

            state = react_nodes.execute_tools_node(state)
            yield {
                "stage": "execute_tools",
                "message": "Executing diagnostic tools",
                "snapshot": self._state_snapshot(state)
            }

            state = react_nodes.reflect_node(state)
            yield {
                "stage": "reflect",
                "message": "Evaluating evidence and refining hypotheses",
                "snapshot": self._state_snapshot(state)
            }

            state = react_nodes.update_memory_node(state)
            yield {
                "stage": "update_memory",
                "message": "Updating clinical memory with case insights",
                "snapshot": self._state_snapshot(state)
            }

            # Loop back to observe
            state = react_nodes.observe_node(state)
            yield {
                "stage": "observe",
                "message": "Re-analyzing with updated context",
                "snapshot": self._state_snapshot(state)
            }

            branch = should_continue_condition(state)
            if branch == "conclude":
                state = react_nodes.conclude_node(state)
                final = self._final_result_from_state(state)
                yield {
                    "stage": "conclude",
                    "message": "Finalizing diagnosis and recommendations",
                    "snapshot": self._state_snapshot(state),
                    "final_result": final
                }
                return
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow structure."""
        return {
            "workflow_type": "ReAct-Style Diagnostic Reasoning",
            "nodes": [
                "observe",
                "retrieve_memory", 
                "think",
                "select_tools",
                "execute_tools",
                "reflect",
                "update_memory",
                "conclude"
            ],
            "features": [
                "Iterative reasoning with multiple cycles",
                "Dynamic tool selection and execution",
                "Clinical memory and pattern recognition",
                "Evidence-based confidence scoring",
                "Structured diagnostic reasoning chain",
                "Adaptive workflow based on diagnostic complexity"
            ],
            "tools_available": [
                "differential_diagnosis_tool",
                "clinical_guidelines_tool", 
                "clinical_decision_rules_tool",
                "medical_knowledge_lookup_tool",
                "severity_assessment_tool"
            ]
        }

# Legacy compatibility function
def run_react_analysis(image_path: str, symptoms: str = "", **kwargs) -> Dict[str, Any]:
    """
    Legacy function for running ReAct analysis (for backward compatibility).
    
    Args:
        image_path: Path to the X-ray image
        symptoms: Patient symptoms
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing analysis results
    """
    runner = ReActDiagnosticWorkflowRunner()
    return runner.run_diagnostic_analysis(image_path, symptoms, **kwargs) 