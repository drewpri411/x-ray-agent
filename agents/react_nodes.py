"""
ReAct Nodes for Diagnostic Reasoning Workflow
"""

from typing import Dict, Any, List
import logging
from agents.diagnostic_state import DiagnosticState
from agents.diagnostic_tools import (
    differential_diagnosis_tool,
    clinical_guidelines_tool,
    clinical_decision_rules_tool,
    medical_knowledge_lookup_tool,
    severity_assessment_tool
)
from agents.memory_system import ClinicalMemorySystem
from agents.image_agent import ImageAnalysisAgent
from agents.symptom_agent import SymptomAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import json

logger = logging.getLogger(__name__)

class ReActNodes:
    """ReAct nodes for diagnostic reasoning workflow."""
    
    def __init__(self):
        """Initialize ReAct nodes."""
        self.image_agent = ImageAnalysisAgent()
        self.symptom_agent = SymptomAgent()
        self.memory_system = ClinicalMemorySystem()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
    
    def observe_node(self, state: DiagnosticState) -> DiagnosticState:
        """Observe: Analyze image findings and symptoms."""
        logger.info(f"🔍 OBSERVE: Analyzing image and symptoms (iteration {state.iteration_count})")
        
        try:
            # Analyze image only if not already analyzed (e.g., precomputed for streaming with heatmaps)
            if not state.image_findings:
                image_results = self.image_agent.analyze_image(state.image_path, generate_heatmaps=False)
                state.image_findings = image_results
            
            # Parse symptoms
            if state.symptoms:
                parsed_symptoms = self.symptom_agent.parse_symptoms(state.symptoms)
                state.parsed_symptoms = parsed_symptoms
            else:
                state.parsed_symptoms = {"primary_symptoms": [], "severity_classification": "unknown"}
            
            # Extract clinical observations
            clinical_observations = []
            
            # Add image findings
            key_findings = state.image_findings.get('key_findings', {})
            for pathology, score in key_findings.items():
                if score > 0.5:
                    clinical_observations.append(f"Moderate {pathology} detected (score: {score:.2f})")
                elif score > 0.3:
                    clinical_observations.append(f"Mild {pathology} suspected (score: {score:.2f})")
            
            # Add symptoms
            if state.parsed_symptoms.get('primary_symptoms'):
                clinical_observations.extend(state.parsed_symptoms['primary_symptoms'])
            
            state.clinical_observations = clinical_observations
            
            # Add reasoning step
            state.add_reasoning_step("observe", {
                "image_findings": key_findings,
                "clinical_observations": clinical_observations,
                "symptoms_parsed": bool(state.symptoms)
            })
            
            logger.info(f"✅ OBSERVE: Found {len(clinical_observations)} clinical observations")
            
        except Exception as e:
            logger.error(f"❌ OBSERVE: Error in observation: {e}")
            state.errors.append(f"Observation error: {str(e)}")
        
        return state
    
    def retrieve_memory_node(self, state: DiagnosticState) -> DiagnosticState:
        """Retrieve relevant clinical memory and patterns."""
        logger.info(f"🧠 MEMORY: Retrieving relevant clinical memory")
        
        try:
            # Search for similar cases
            query_data = {
                "findings": state.image_findings.get('key_findings', {}),
                "symptoms": state.parsed_symptoms.get('primary_symptoms', [])
            }
            
            similar_cases = self.memory_system.search_similar_cases(query_data, top_k=3)
            state.similar_cases = similar_cases
            
            # Get memory insights
            memory_insights = []
            for case in similar_cases:
                memory_insights.append({
                    'case_id': case['id'],
                    'similarity': case.get('similarity', 0.0),
                    'diagnosis': case.get('diagnosis', ''),
                    'confidence': case.get('confidence', 0.0),
                    'severity': case.get('severity', ''),
                    'learnings': case.get('learnings', [])
                })
            
            state.clinical_memory = memory_insights
            
            # Add reasoning step
            state.add_reasoning_step("retrieve_memory", {
                "similar_cases_found": len(similar_cases),
                "memory_insights": memory_insights
            })
            
            logger.info(f"✅ MEMORY: Retrieved {len(similar_cases)} similar cases")
            
        except Exception as e:
            logger.error(f"❌ MEMORY: Error retrieving memory: {e}")
            state.errors.append(f"Memory retrieval error: {str(e)}")
        
        return state
    
    def think_node(self, state: DiagnosticState) -> DiagnosticState:
        """Think: Generate diagnostic hypotheses and reasoning."""
        logger.info(f"🤔 THINK: Generating diagnostic hypotheses (iteration {state.iteration_count})")
        
        try:
            # Create diagnostic reasoning prompt
            prompt = self._create_diagnostic_prompt(state)
            
            # Use LLM for diagnostic reasoning
            response = self.llm.invoke(prompt)
            
            # Parse structured response
            reasoning_result = self._parse_diagnostic_reasoning(response.content)
            
            # Update hypotheses
            for hypothesis in reasoning_result.get('hypotheses', []):
                state.update_hypothesis(
                    diagnosis=hypothesis['diagnosis'],
                    confidence=hypothesis['confidence'],
                    evidence=hypothesis['evidence'],
                    contradicting_evidence=hypothesis.get('contradicting_evidence', [])
                )
            
            # Add reasoning step
            state.add_reasoning_step("think", {
                "hypotheses_generated": len(reasoning_result.get('hypotheses', [])),
                "reasoning": reasoning_result.get('reasoning', ''),
                "confidence_levels": reasoning_result.get('confidence_levels', {})
            })
            
            logger.info(f"✅ THINK: Generated {len(reasoning_result.get('hypotheses', []))} hypotheses")
            
        except Exception as e:
            logger.error(f"❌ THINK: Error in reasoning: {e}")
            state.errors.append(f"Reasoning error: {str(e)}")
        
        return state
    
    def select_tools_node(self, state: DiagnosticState) -> DiagnosticState:
        """Select which diagnostic tools to use based on current hypotheses."""
        logger.info(f"🛠️ SELECT TOOLS: Choosing diagnostic tools")
        
        try:
            selected_tools = []
            
            # Always include core diagnostic tools
            selected_tools.extend([
                "differential_diagnosis_tool",
                "severity_assessment_tool",
                "medical_knowledge_lookup_tool"
            ])
            
            # Select additional tools based on top hypotheses
            top_hypothesis = state.get_top_hypothesis()
            if top_hypothesis:
                diagnosis = top_hypothesis['diagnosis'].lower()
                
                if "pneumonia" in diagnosis:
                    selected_tools.extend([
                        "clinical_guidelines_tool",
                        "clinical_decision_rules_tool"
                    ])
                elif "heart" in diagnosis or "failure" in diagnosis:
                    selected_tools.extend([
                        "clinical_guidelines_tool",
                        "clinical_decision_rules_tool"
                    ])
                elif "effusion" in diagnosis:
                    selected_tools.append("clinical_guidelines_tool")
                elif "atelectasis" in diagnosis:
                    selected_tools.extend([
                        "medical_knowledge_lookup_tool",
                        "clinical_guidelines_tool"
                    ])
            
            state.selected_tools = selected_tools
            
            # Add reasoning step
            state.add_reasoning_step("select_tools", {
                "tools_selected": selected_tools,
                "reasoning": f"Selected tools based on top hypothesis: {top_hypothesis['diagnosis'] if top_hypothesis else 'None'}"
            })
            
            logger.info(f"✅ SELECT TOOLS: Selected {len(selected_tools)} tools")
            
        except Exception as e:
            logger.error(f"❌ SELECT TOOLS: Error selecting tools: {e}")
            state.errors.append(f"Tool selection error: {str(e)}")
        
        return state
    
    def execute_tools_node(self, state: DiagnosticState) -> DiagnosticState:
        """Execute selected diagnostic tools."""
        logger.info(f"⚡ EXECUTE TOOLS: Running diagnostic tools")
        
        try:
            tool_results = {}
            
            for tool_name in state.selected_tools:
                try:
                    if tool_name == "differential_diagnosis_tool":
                        # Convert list to dict for tool input
                        observations_dict = {"observations": state.clinical_observations}
                        try:
                            result = differential_diagnosis_tool(observations_dict)
                            tool_results[tool_name] = result
                        except Exception as e:
                            logger.warning(f"Tool {tool_name} failed: {e}")
                            tool_results[tool_name] = {"error": str(e)}
                        
                    elif tool_name == "clinical_guidelines_tool":
                        top_hypothesis = state.get_top_hypothesis()
                        if top_hypothesis:
                            try:
                                result = clinical_guidelines_tool(
                                    diagnosis=top_hypothesis['diagnosis'],
                                    severity="moderate"
                                )
                                tool_results[tool_name] = result
                            except Exception as e:
                                logger.warning(f"Tool {tool_name} failed: {e}")
                                tool_results[tool_name] = {"error": str(e)}
                    
                    elif tool_name == "clinical_decision_rules_tool":
                        # Create patient data for decision rules
                        patient_data = {
                            "symptoms": state.symptoms,
                            "age": state.patient_info.get("age", 50),
                            "confusion": "confusion" in state.symptoms.lower(),
                            "respiratory_rate": state.patient_info.get("respiratory_rate", 20)
                        }
                        
                        top_hypothesis = state.get_top_hypothesis()
                        if top_hypothesis and "pneumonia" in top_hypothesis['diagnosis'].lower():
                            try:
                                result = clinical_decision_rules_tool("curb65", patient_data)
                                tool_results[tool_name] = result
                            except Exception as e:
                                logger.warning(f"Tool {tool_name} failed: {e}")
                                tool_results[tool_name] = {"error": str(e)}
                    
                    elif tool_name == "medical_knowledge_lookup_tool":
                        top_hypothesis = state.get_top_hypothesis()
                        if top_hypothesis:
                            try:
                                result = medical_knowledge_lookup_tool(top_hypothesis['diagnosis'])
                                tool_results[tool_name] = result
                            except Exception as e:
                                logger.warning(f"Tool {tool_name} failed: {e}")
                                tool_results[tool_name] = {"error": str(e)}
                    
                    elif tool_name == "severity_assessment_tool":
                        try:
                            result = severity_assessment_tool(
                                image_findings=state.image_findings,
                                symptoms=state.symptoms
                            )
                            tool_results[tool_name] = result
                        except Exception as e:
                            logger.warning(f"Tool {tool_name} failed: {e}")
                            tool_results[tool_name] = {"error": str(e)}
                    
                    # Track tool usage
                    state.add_tool_usage(tool_name, {}, tool_results[tool_name])
                    
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
            
            state.tool_results = tool_results
            
            # Add reasoning step
            state.add_reasoning_step("execute_tools", {
                "tools_executed": len(tool_results),
                "successful_tools": [name for name, result in tool_results.items() if "error" not in result],
                "failed_tools": [name for name, result in tool_results.items() if "error" in result]
            })
            
            logger.info(f"✅ EXECUTE TOOLS: Executed {len(tool_results)} tools")
            
        except Exception as e:
            logger.error(f"❌ EXECUTE TOOLS: Error executing tools: {e}")
            state.errors.append(f"Tool execution error: {str(e)}")
        
        return state
    
    def reflect_node(self, state: DiagnosticState) -> DiagnosticState:
        """Reflect: Evaluate evidence and refine hypotheses."""
        logger.info(f"🔄 REFLECT: Evaluating evidence and refining hypotheses (iteration {state.iteration_count})")
        
        try:
            # Integrate tool results with current hypotheses
            updated_hypotheses = []
            
            for hypothesis in state.hypotheses:
                # Gather all evidence for this hypothesis
                supporting_evidence = state.evidence_for.get(hypothesis['diagnosis'], [])
                contradicting_evidence = state.evidence_against.get(hypothesis['diagnosis'], [])
                
                # Add tool-based evidence
                for tool_name, tool_result in state.tool_results.items():
                    if "error" not in tool_result:
                        if tool_name == "differential_diagnosis_tool":
                            for diff in tool_result.get("differential_diagnoses", []):
                                if diff["diagnosis"].lower() == hypothesis['diagnosis'].lower():
                                    supporting_evidence.append(f"Tool evidence: {diff['supporting_evidence']}")
                        
                        elif tool_name == "clinical_guidelines_tool":
                            if tool_result.get("diagnosis", "").lower() == hypothesis['diagnosis'].lower():
                                supporting_evidence.append(f"Guidelines support: {tool_result.get('recommended_treatment', [])}")
                        
                        elif tool_name == "severity_assessment_tool":
                            severity = tool_result.get("severity_category", "")
                            supporting_evidence.append(f"Severity assessment: {severity}")
                
                # Recalculate confidence based on all evidence
                new_confidence = self._calculate_confidence(
                    supporting_evidence, 
                    contradicting_evidence,
                    hypothesis['confidence']
                )
                
                updated_hypotheses.append({
                    **hypothesis,
                    'confidence': new_confidence,
                    'supporting_evidence': supporting_evidence,
                    'contradicting_evidence': contradicting_evidence
                })
            
            state.hypotheses = updated_hypotheses
            
            # Determine if we need more iterations
            if state.should_stop_reasoning():
                state.should_continue = False
            
            state.iteration_count += 1
            
            # Add reasoning step
            state.add_reasoning_step("reflect", {
                "hypotheses_updated": len(updated_hypotheses),
                "should_continue": state.should_continue,
                "iteration_count": state.iteration_count
            })
            
            logger.info(f"✅ REFLECT: Updated {len(updated_hypotheses)} hypotheses, continue: {state.should_continue}")
            
        except Exception as e:
            logger.error(f"❌ REFLECT: Error in reflection: {e}")
            state.errors.append(f"Reflection error: {str(e)}")
        
        return state
    
    def update_memory_node(self, state: DiagnosticState) -> DiagnosticState:
        """Update clinical memory with current case insights."""
        logger.info(f"💾 UPDATE MEMORY: Storing case insights")
        
        try:
            # Extract learnings from this iteration
            top_hypothesis = state.get_top_hypothesis()
            
            case_data = {
                "findings": state.image_findings.get('key_findings', {}),
                "symptoms": state.parsed_symptoms.get('primary_symptoms', []),
                "diagnosis": top_hypothesis['diagnosis'] if top_hypothesis else "unknown",
                "confidence": top_hypothesis['confidence'] if top_hypothesis else 0.0,
                "severity": "moderate",  # Could be enhanced with severity assessment
                "learnings": [
                    f"Iteration {state.iteration_count} insights",
                    f"Tools used: {', '.join(state.selected_tools)}",
                    f"Final confidence: {top_hypothesis['confidence'] if top_hypothesis else 0.0}"
                ]
            }
            
            # Store in memory
            case_id = self.memory_system.add_case(case_data)
            
            # Add reasoning step
            state.add_reasoning_step("update_memory", {
                "case_stored": case_id,
                "learnings": case_data["learnings"]
            })
            
            logger.info(f"✅ UPDATE MEMORY: Stored case {case_id}")
            
        except Exception as e:
            logger.error(f"❌ UPDATE MEMORY: Error updating memory: {e}")
            state.errors.append(f"Memory update error: {str(e)}")
        
        return state
    
    def conclude_node(self, state: DiagnosticState) -> DiagnosticState:
        """Conclude: Provide final diagnosis and recommendations."""
        logger.info(f"🎯 CONCLUDE: Finalizing diagnosis and recommendations")
        
        try:
            # Get top hypothesis
            top_hypothesis = state.get_top_hypothesis()
            
            if top_hypothesis:
                state.primary_diagnosis = top_hypothesis['diagnosis']
                state.confidence_scores[top_hypothesis['diagnosis']] = top_hypothesis['confidence']
                
                # Create differential diagnoses
                state.differential_diagnoses = []
                for hypothesis in state.hypotheses[:5]:  # Top 5
                    if hypothesis['diagnosis'] != top_hypothesis['diagnosis']:
                        state.differential_diagnoses.append({
                            'diagnosis': hypothesis['diagnosis'],
                            'confidence': hypothesis['confidence'],
                            'evidence': hypothesis['evidence']
                        })
                
                # Determine triage category based on confidence and severity
                severity_result = state.tool_results.get("severity_assessment_tool", {})
                severity_category = severity_result.get("severity_category", "moderate")
                
                # Calculate urgency score based on confidence and findings
                urgency_score = 0.0
                
                # Base urgency from confidence
                if top_hypothesis['confidence'] > 0.8:
                    urgency_score += 0.4
                elif top_hypothesis['confidence'] > 0.6:
                    urgency_score += 0.3
                elif top_hypothesis['confidence'] > 0.4:
                    urgency_score += 0.2
                else:
                    urgency_score += 0.1
                
                # Add urgency from image findings
                key_findings = state.image_findings.get('key_findings', {})
                high_urgency_pathologies = ['pneumonia', 'effusion', 'cardiomegaly', 'edema', 'consolidation']
                for pathology in high_urgency_pathologies:
                    if pathology in key_findings:
                        score = key_findings[pathology]
                        if score > 0.7:
                            urgency_score += 0.3
                        elif score > 0.5:
                            urgency_score += 0.2
                        elif score > 0.3:
                            urgency_score += 0.1
                
                # Add urgency from symptoms
                if state.symptoms and any(word in state.symptoms.lower() for word in ['severe', 'acute', 'emergency', 'unable to breathe']):
                    urgency_score += 0.2
                
                # Cap urgency score at 1.0
                urgency_score = min(urgency_score, 1.0)
                
                # Determine triage category
                if urgency_score >= 0.8:
                    state.triage_category = "EMERGENCY"
                elif urgency_score >= 0.6:
                    state.triage_category = "URGENT"
                elif urgency_score >= 0.4:
                    state.triage_category = "MODERATE"
                else:
                    state.triage_category = "ROUTINE"
                
                state.urgency_score = urgency_score
                
                # Generate final recommendations
                state.final_recommendations = self._generate_final_recommendations(
                    state, top_hypothesis, severity_result
                )
            
            # Set workflow status
            state.workflow_status = "completed"
            
            # Add reasoning step
            state.add_reasoning_step("conclude", {
                "primary_diagnosis": state.primary_diagnosis,
                "triage_category": state.triage_category,
                "urgency_score": state.urgency_score,
                "recommendations_count": len(state.final_recommendations)
            })
            
            logger.info(f"✅ CONCLUDE: Final diagnosis: {state.primary_diagnosis} ({state.triage_category})")
            
        except Exception as e:
            logger.error(f"❌ CONCLUDE: Error in conclusion: {e}")
            state.errors.append(f"Conclusion error: {str(e)}")
            state.workflow_status = "failed"
        
        return state
    
    def _create_diagnostic_prompt(self, state: DiagnosticState) -> str:
        """Create prompt for diagnostic reasoning."""
        # Convert numpy values to regular Python types for JSON serialization
        key_findings = {}
        for k, v in state.image_findings.get('key_findings', {}).items():
            key_findings[k] = float(v) if hasattr(v, 'item') else v
        
        # Convert similar cases for JSON serialization
        similar_cases_serializable = []
        for case in state.similar_cases:
            case_copy = {}
            for k, v in case.items():
                if hasattr(v, 'item'):  # numpy type
                    case_copy[k] = float(v)
                elif isinstance(v, (list, dict)):
                    case_copy[k] = self._convert_to_serializable(v)
                else:
                    case_copy[k] = v
            similar_cases_serializable.append(case_copy)
        
        # Convert reasoning chain for JSON serialization
        reasoning_chain_serializable = []
        for step in state.reasoning_chain[-3:] if state.reasoning_chain else []:
            step_copy = {}
            for k, v in step.items():
                if hasattr(v, 'item'):  # numpy type
                    step_copy[k] = float(v)
                elif isinstance(v, (list, dict)):
                    step_copy[k] = self._convert_to_serializable(v)
                else:
                    step_copy[k] = v
            reasoning_chain_serializable.append(step_copy)
        
        return f"""
        You are an expert radiologist performing diagnostic reasoning. Based on the following clinical evidence, generate diagnostic hypotheses:

        IMAGE FINDINGS:
        {json.dumps(key_findings, indent=2)}

        CLINICAL OBSERVATIONS:
        {state.clinical_observations}

        SYMPTOMS:
        {state.symptoms if state.symptoms else "No symptoms provided"}

        SIMILAR CASES FROM MEMORY:
        {json.dumps(similar_cases_serializable, indent=2)}

        PREVIOUS REASONING:
        {json.dumps(reasoning_chain_serializable, indent=2)}

        IMPORTANT DIAGNOSTIC GUIDELINES:
        - ATELECTASIS: Look for volume loss, displacement of fissures, elevated hemidiaphragm, mediastinal shift
        - PNEUMONIA: Look for consolidation, air bronchograms, lobar or segmental opacities
        - PLEURAL EFFUSION: Look for blunting of costophrenic angles, meniscus sign, mediastinal shift
        - CARDIOMEGALY: Look for enlarged cardiac silhouette (>50% of thoracic width)
        - PULMONARY EDEMA: Look for perihilar opacities, Kerley B lines, bat wing appearance
        - CONSOLIDATION: Look for homogeneous opacification, air bronchograms

        CRITICAL: Pay close attention to the specific findings and their patterns. Do not confuse atelectasis with other conditions. Atelectasis typically shows volume loss and displacement, not consolidation or edema patterns.

        Generate a structured response with:
        1. Primary hypothesis with confidence (0-100%)
        2. 3-5 differential diagnoses with confidence
        3. Key evidence supporting each hypothesis
        4. Evidence against each hypothesis
        5. Clinical reasoning for your conclusions

        Format your response as JSON:
        {{
            "hypotheses": [
                {{
                    "diagnosis": "condition_name",
                    "confidence": 0.85,
                    "evidence": ["evidence1", "evidence2"],
                    "contradicting_evidence": ["contradicting1"]
                }}
            ],
            "reasoning": "Clinical reasoning explanation",
            "confidence_levels": {{
                "high": ">0.8",
                "moderate": "0.6-0.8", 
                "low": "<0.6"
            }}
        }}
        """
    
    def _parse_diagnostic_reasoning(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {
                    "hypotheses": [],
                    "reasoning": response,
                    "confidence_levels": {}
                }
        except Exception as e:
            logger.warning(f"Failed to parse diagnostic reasoning: {e}")
            return {
                "hypotheses": [],
                "reasoning": response,
                "confidence_levels": {}
            }
    
    def _calculate_confidence(self, supporting_evidence: List[str], 
                            contradicting_evidence: List[str], 
                            base_confidence: float) -> float:
        """Calculate confidence based on evidence."""
        # Base confidence
        confidence = base_confidence
        
        # Adjust based on evidence
        confidence += len(supporting_evidence) * 0.05
        confidence -= len(contradicting_evidence) * 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _generate_final_recommendations(self, state: DiagnosticState, 
                                      top_hypothesis: Dict[str, Any], 
                                      severity_result: Dict[str, Any]) -> List[str]:
        """Generate final clinical recommendations."""
        recommendations = []
        
        # Diagnosis-specific recommendations
        diagnosis = top_hypothesis['diagnosis'].lower()
        confidence = top_hypothesis['confidence']
        
        if "pneumonia" in diagnosis:
            if confidence > 0.8:
                recommendations.append("🔴 HIGH PROBABILITY PNEUMONIA: Immediate antibiotic therapy recommended")
            else:
                recommendations.append("🟡 SUSPECTED PNEUMONIA: Consider antibiotic therapy based on clinical assessment")
        
        elif "heart" in diagnosis or "failure" in diagnosis:
            recommendations.append("🫀 CARDIAC EVALUATION: Echocardiography and cardiac biomarkers recommended")
        
        elif "effusion" in diagnosis:
            recommendations.append("💧 PLEURAL EFFUSION: Thoracentesis may be indicated for diagnosis")
        
        # Severity-based recommendations
        severity = severity_result.get("severity_category", "moderate")
        if severity == "severe":
            recommendations.append("🚨 SEVERE CASE: Immediate medical attention required")
        elif severity == "moderate":
            recommendations.append("⚠️ MODERATE SEVERITY: Prompt evaluation recommended")
        
        # General recommendations
        recommendations.append("📋 FOLLOW-UP: Schedule appropriate follow-up imaging and clinical assessment")
        recommendations.append("🔬 ADDITIONAL TESTS: Consider relevant laboratory studies based on differential diagnosis")
        
        return recommendations
    
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