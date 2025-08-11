"""
LangGraph Workflow for Radiology Assistant
Orchestrates the complete workflow from image upload to report generation.
"""

from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import logging
from agents.symptom_agent import SymptomAgent
from agents.image_agent import ImageAnalysisAgent
from agents.triage_agent import TriageAgent
from agents.report_agent import ReportAgent
import os

logger = logging.getLogger(__name__)

# Define the state structure
class RadiologyState(TypedDict):
    """State structure for the radiology workflow."""
    image_path: str
    symptoms: str
    parsed_symptoms: Dict[str, Any]
    image_analysis: Dict[str, Any]
    triage_result: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    workflow_status: str

def create_radiology_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for radiology analysis.
    
    Returns:
        Configured StateGraph for the workflow
    """
    
    # Initialize agents
    symptom_agent = SymptomAgent()
    image_agent = ImageAnalysisAgent()
    triage_agent = TriageAgent()
    report_agent = ReportAgent()
    
    # Define workflow nodes
    def parse_symptoms_node(state: RadiologyState) -> RadiologyState:
        """Parse and categorize patient symptoms."""
        try:
            logger.info("Starting symptom parsing")
            symptoms = state.get('symptoms', '')
            
            if symptoms.strip():
                parsed_symptoms = symptom_agent.parse_symptoms(symptoms)
                state['parsed_symptoms'] = parsed_symptoms
                logger.info("Symptom parsing completed")
            else:
                state['parsed_symptoms'] = {
                    'primary_symptoms': [],
                    'severity_classification': 'low',
                    'urgency_level': 'low'
                }
                state['warnings'].append("No symptoms provided")
            
            return state
            
        except Exception as e:
            logger.error(f"Symptom parsing failed: {e}")
            state['errors'].append(f"Symptom parsing error: {str(e)}")
            state['workflow_status'] = 'error'
            return state
    
    def analyze_image_node(state: RadiologyState) -> RadiologyState:
        """Analyze the X-ray image."""
        try:
            logger.info("Starting image analysis")
            image_path = state.get('image_path', '')
            
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate image
            validation = image_agent.validate_image(image_path)
            if not validation['is_valid']:
                raise ValueError(f"Image validation failed: {validation['errors']}")
            
            # Get heatmap preference from state
            generate_heatmaps = state.get('generate_heatmaps', False)
            
            # Perform analysis
            image_analysis = image_agent.analyze_image(image_path, generate_heatmaps=generate_heatmaps)
            state['image_analysis'] = image_analysis
            
            if image_analysis.get('analysis_status') == 'failed':
                raise Exception(f"Image analysis failed: {image_analysis.get('error', 'Unknown error')}")
            
            logger.info("Image analysis completed")
            return state
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            state['errors'].append(f"Image analysis error: {str(e)}")
            state['workflow_status'] = 'error'
            return state
    
    def triage_assessment_node(state: RadiologyState) -> RadiologyState:
        """Perform triage assessment."""
        try:
            logger.info("Starting triage assessment")
            
            image_analysis = state.get('image_analysis', {})
            symptoms = state.get('symptoms', '')
            parsed_symptoms = state.get('parsed_symptoms', {})
            
            if not image_analysis:
                raise ValueError("No image analysis results available for triage")
            
            # Perform triage assessment
            triage_result = triage_agent.assess_urgency(
                image_analysis=image_analysis,
                symptoms=symptoms,
                parsed_symptoms=parsed_symptoms
            )
            
            state['triage_result'] = triage_result
            logger.info(f"Triage assessment completed: {triage_result.get('triage_category', 'UNKNOWN')}")
            return state
            
        except Exception as e:
            logger.error(f"Triage assessment failed: {e}")
            state['errors'].append(f"Triage assessment error: {str(e)}")
            state['workflow_status'] = 'error'
            return state
    
    def generate_report_node(state: RadiologyState) -> RadiologyState:
        """Generate comprehensive medical report."""
        try:
            logger.info("Starting report generation")
            
            image_analysis = state.get('image_analysis', {})
            triage_result = state.get('triage_result', {})
            symptoms = state.get('symptoms', '')
            
            if not image_analysis or not triage_result:
                raise ValueError("Missing required data for report generation")
            
            # Generate report
            report = report_agent.generate_report(
                image_analysis=image_analysis,
                triage_result=triage_result,
                symptoms=symptoms
            )
            
            state['report'] = report
            state['workflow_status'] = 'completed'
            logger.info("Report generation completed")
            return state
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state['errors'].append(f"Report generation error: {str(e)}")
            state['workflow_status'] = 'error'
            return state
    
    def error_handler_node(state: RadiologyState) -> RadiologyState:
        """Handle errors in the workflow."""
        logger.error(f"Workflow error: {state.get('errors', [])}")
        state['workflow_status'] = 'error'
        return state
    
    # Create the workflow graph
    workflow = StateGraph(RadiologyState)
    
    # Add nodes
    workflow.add_node("parse_symptoms", parse_symptoms_node)
    workflow.add_node("analyze_image", analyze_image_node)
    workflow.add_node("triage_assessment", triage_assessment_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define the workflow edges
    workflow.set_entry_point("parse_symptoms")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "parse_symptoms",
        lambda state: "analyze_image" if state.get('workflow_status') != 'error' else "error_handler"
    )
    
    workflow.add_conditional_edges(
        "analyze_image",
        lambda state: "triage_assessment" if state.get('workflow_status') != 'error' else "error_handler"
    )
    
    workflow.add_conditional_edges(
        "triage_assessment",
        lambda state: "generate_report" if state.get('workflow_status') != 'error' else "error_handler"
    )
    
    workflow.add_conditional_edges(
        "generate_report",
        lambda state: END if state.get('workflow_status') != 'error' else "error_handler"
    )
    
    workflow.add_conditional_edges(
        "error_handler",
        lambda state: END
    )
    
    return workflow.compile()

class RadiologyWorkflowRunner:
    """Runner class for the radiology workflow."""
    
    def __init__(self):
        """Initialize the workflow runner."""
        self.workflow = create_radiology_workflow()
        logger.info("Radiology workflow initialized")
    
    def run_analysis(self, 
                    image_path: str, 
                    symptoms: str = "",
                    generate_heatmaps: bool = False) -> Dict[str, Any]:
        """
        Run the complete radiology analysis workflow.
        
        Args:
            image_path: Path to the X-ray image file
            symptoms: Patient symptoms text
            
        Returns:
            Complete analysis results
        """
        try:
            # Initialize state
            initial_state = RadiologyState(
                image_path=image_path,
                symptoms=symptoms,
                parsed_symptoms={},
                image_analysis={},
                triage_result={},
                report={},
                errors=[],
                warnings=[],
                workflow_status='running'
            )
            
            # Store heatmap preference in state
            initial_state['generate_heatmaps'] = generate_heatmaps
            
            logger.info(f"Starting radiology analysis for: {image_path}")
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            # Compile final results
            final_results = {
                'workflow_status': result.get('workflow_status', 'unknown'),
                'image_path': image_path,
                'symptoms': symptoms,
                'parsed_symptoms': result.get('parsed_symptoms', {}),
                'image_analysis': result.get('image_analysis', {}),
                'triage_result': result.get('triage_result', {}),
                'report': result.get('report', {}),
                'errors': result.get('errors', []),
                'warnings': result.get('warnings', []),
                'summary': self._generate_summary(result)
            }
            
            logger.info(f"Radiology analysis completed with status: {final_results['workflow_status']}")
            return final_results
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'workflow_status': 'error',
                'image_path': image_path,
                'symptoms': symptoms,
                'errors': [f"Workflow execution error: {str(e)}"],
                'warnings': [],
                'summary': f"Analysis failed: {str(e)}"
            }
    
    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a summary of the analysis results."""
        try:
            workflow_status = result.get('workflow_status', 'unknown')
            
            if workflow_status == 'completed':
                triage_category = result.get('triage_result', {}).get('triage_category', 'UNKNOWN')
                urgency_score = result.get('triage_result', {}).get('urgency_score', 0.0)
                
                summary = f"Analysis completed successfully. "
                summary += f"Triage category: {triage_category} (urgency score: {urgency_score:.3f})"
                
                # Add key findings
                key_findings = result.get('image_analysis', {}).get('key_findings', {})
                if key_findings:
                    findings_text = ', '.join([f"{k}: {v:.3f}" for k, v in key_findings.items()])
                    summary += f" Key findings: {findings_text}"
                
                return summary
                
            elif workflow_status == 'error':
                errors = result.get('errors', [])
                return f"Analysis failed with errors: {'; '.join(errors)}"
            
            else:
                return f"Analysis status: {workflow_status}"
                
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def validate_inputs(self, image_path: str, symptoms: str = "") -> Dict[str, Any]:
        """
        Validate inputs before running the workflow.
        
        Args:
            image_path: Path to the X-ray image file
            symptoms: Patient symptoms text
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate image path
        if not image_path:
            validation_result['errors'].append("No image path provided")
            validation_result['is_valid'] = False
        elif not os.path.exists(image_path):
            validation_result['errors'].append(f"Image file not found: {image_path}")
            validation_result['is_valid'] = False
        
        # Validate symptoms (optional)
        if not symptoms or symptoms.strip() == "":
            validation_result['warnings'].append("No symptoms provided - analysis will be image-only")
        
        return validation_result 