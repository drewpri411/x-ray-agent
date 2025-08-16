"""
Streamlit Web Interface for Radiology Assistant
Provides a user-friendly web interface for chest X-ray analysis.
"""

import streamlit as st
import os
import sys
import tempfile
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import RadiologyWorkflowRunner
from workflow_react import ReActDiagnosticWorkflowRunner
from utils.helpers import setup_logging, save_json

# Load environment variables from .env file
load_dotenv()

# Setup logging
setup_logging()

def check_environment_setup():
    """Check if environment is properly configured."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    env_file_exists = os.path.exists(".env")
    
    return {
        "api_key_available": bool(google_api_key),
        "env_file_exists": env_file_exists,
        "setup_instructions": not env_file_exists
    }

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Radiology Assistant",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🫁 AI Radiology Assistant</h1>', unsafe_allow_html=True)
    
    # Warning disclaimer
    # st.markdown("""
    # <div class="warning-box">
    #     <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
    #     This is a prototype AI system for educational and demonstration purposes only. 
    #     All findings should be reviewed by qualified healthcare professionals before clinical use.
    #     This system is NOT intended for actual medical diagnosis or treatment decisions.
    # </div>
    # """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Workflow selection
        use_react = st.checkbox("🧠 Use ReAct Diagnostic Reasoning", value=False, 
                               help="Enable advanced ReAct-style diagnostic reasoning with iterative analysis, tools, and memory")
        
        if use_react:
            st.success("✅ ReAct Workflow Selected")
            st.info("Features: Iterative reasoning, tools, memory, evidence tracking")
        else:
            st.info("ℹ️ Legacy Workflow Selected")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["densenet121-res224-all", "densenet121-res224-chex", "densenet121-res224-mimic_ch"],
            help="Choose the TorchXRayVision model for analysis"
        )
        
                # Analysis options
        st.subheader("Analysis Options")
        
        # Check if Grad-CAM is available
        try:
            from models.grad_cam_tool import XRayGradCAM, GRADCAM_AVAILABLE
            if GRADCAM_AVAILABLE:
                from models.image_model import ChestXRayClassifier
                test_classifier = ChestXRayClassifier()
                test_gradcam = XRayGradCAM(test_classifier.model)
                gradcam_available = True
            else:
                gradcam_available = False
        except Exception as e:
            gradcam_available = False
        
        if gradcam_available:
            generate_heatmaps = st.checkbox("Generate Heatmaps", value=True, help="Generate Grad-CAM heatmaps for explainability")
        else:
            st.warning("⚠️ Grad-CAM not available - heatmaps disabled")
            generate_heatmaps = False
        
        save_results = st.checkbox("Save Results", value=False, help="Save analysis results to file")
        
        # Environment status
        st.subheader("Environment Status")
        env_status = check_environment_setup()
        
        if env_status["api_key_available"]:
            st.success("✅ Google AI API Key loaded from environment")
            st.info("Full LLM features enabled")
        elif env_status["setup_instructions"]:
            st.error("❌ .env file not found")
            st.markdown("""
            **Setup Required:**
            1. Copy `env.example` to `.env`
            2. Add your Google AI API key to `.env`
            3. Restart the application
            """)
            if st.button("📋 Show Setup Instructions"):
                st.code("""
# In your terminal, run:
cp env.example .env

# Then edit .env and add your API key:
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.5-pro
                """)
        else:
            st.warning("⚠️ Google AI API Key not found in .env file")
            st.info("Basic analysis available (no LLM report generation)")
            st.markdown("""
            **To enable full features:**
            1. Edit `.env` file
            2. Add your Google AI API key
            3. Restart the application
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        # Show API key status in main area
        env_status = check_environment_setup()
        if not env_status["api_key_available"]:
            if env_status["setup_instructions"]:
                st.markdown("""
                <div class="error-box">
                    <strong>🔧 Setup Required</strong><br>
                    Please create a `.env` file with your Google AI API key to enable full features.
                    Basic image analysis will still work without the API key.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>🔑 API Key Required for Full Features</strong><br>
                    To enable LLM-based report generation, please add your Google AI API key to the `.env` file.
                    Basic image analysis will still work without the API key.
                </div>
                """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
            help="Upload a chest X-ray image for analysis"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
        
        # Symptoms input
        symptoms = st.text_area(
            "Patient Symptoms (Optional)",
            placeholder="Describe patient symptoms, e.g., 'Patient has cough and fever for 3 days'",
            height=100,
            help="Provide patient symptoms for better triage assessment"
        )
    
    with col2:
        st.header("📊 Results")

        # Initialize session state for process view
        if "analysis_in_progress" not in st.session_state:
            st.session_state.analysis_in_progress = False
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False
        if "last_results" not in st.session_state:
            st.session_state.last_results = None
        if "process_events" not in st.session_state:
            st.session_state.process_events = []

        # Reset helper
        def _reset_analysis():
            st.session_state.analysis_in_progress = False
            st.session_state.analysis_done = False
            st.session_state.last_results = None
            st.session_state.process_events = []
            st.experimental_rerun()

        # Do not return early; we want to keep the stages visible above the final report
        
        # Analysis button
        if uploaded_file is not None and not st.session_state.analysis_in_progress:
            # Show analysis capabilities
            env_status = check_environment_setup()
            if env_status["api_key_available"]:
                st.success("✅ Full analysis available: Image analysis + LLM report generation")
            else:
                st.info("ℹ️ Basic analysis available: Image analysis only (no LLM report)")
            
            if st.button("🚀 Start Analysis", type="primary"):
                st.session_state.analysis_in_progress = True
                st.session_state.process_events = []

        # Create containers for different sections
        report_container = st.container()  # For final report at top
        current_stage_container = st.container()  # For current stage
        completed_stages_container = st.container()  # For completed stages at bottom
        
        # While analysis in progress or if started, show streaming process for ReAct
        if st.session_state.analysis_in_progress:

            try:
                if use_react:
                    runner = ReActDiagnosticWorkflowRunner()
                    st.info("🧠 Using ReAct Diagnostic Reasoning Workflow (live)")
                    
                    final_results = None
                    # Stream stage-by-stage updates
                    for event in runner.run_diagnostic_analysis_stream(temp_image_path, symptoms, generate_heatmaps=generate_heatmaps):
                        # Accumulate events
                        st.session_state.process_events.append(event)
                        
                        # Clear and re-render current stage at top
                        with current_stage_container:
                            st.subheader("🔄 Current Stage")
                            ev = st.session_state.process_events[-1]  # Latest event
                            stage = ev.get('stage','')
                            message = ev.get('message','')
                            snap = ev.get('snapshot') or {}
                            
                            with st.expander(f"🔄 {stage} — {message}", expanded=True):
                                # Stage-specific UI for current stage
                                if stage == 'observe':
                                    cols = st.columns([2, 3])
                                    with cols[0]:
                                        try:
                                            st.image(Image.open(temp_image_path), caption="Input X-ray", use_container_width=True)
                                        except Exception:
                                            pass
                                    with cols[1]:
                                        findings = snap.get('image_findings', {})
                                        if findings:
                                            st.write("Detected findings (preview)")
                                            try:
                                                st.bar_chart({"score": findings})
                                            except Exception:
                                                st.table({"Pathology": list(findings.keys()), "Score": [f"{v:.3f}" if isinstance(v, (int,float)) else v for v in findings.values()]})
                                    # Heatmaps if available from the event
                                    heatmaps = ev.get('heatmaps')
                                    if heatmaps:
                                        st.write("Heatmaps (Grad-CAM)")
                                        for pathology, hm in list(heatmaps.items())[:4]:
                                            if isinstance(hm, dict) and 'figure' in hm:
                                                st.write(f"{pathology}")
                                                st.pyplot(hm['figure'])
                                elif stage == 'retrieve_memory':
                                    cases = snap.get('similar_cases', [])
                                    st.write(f"Similar cases found: {len(cases)}")
                                    if cases:
                                        preview = [{
                                            "id": c.get('id',''),
                                            "diagnosis": c.get('diagnosis',''),
                                            "similarity": f"{c.get('similarity',0):.2f}"
                                        } for c in cases]
                                        st.table({
                                            "Case ID": [p['id'] for p in preview],
                                            "Diagnosis": [p['diagnosis'] for p in preview],
                                            "Similarity": [p['similarity'] for p in preview]
                                        })
                                elif stage == 'think':
                                    hyps = snap.get('hypotheses', [])
                                    if hyps:
                                        for i, h in enumerate(hyps[:5], start=1):
                                            st.write(f"{i}. {h.get('diagnosis','')} (confidence: {h.get('confidence',0):.2f})")
                                    else:
                                        st.write("No hypotheses generated yet")
                                elif stage == 'select_tools':
                                    tools = snap.get('selected_tools', [])
                                    if tools:
                                        st.write("Tools selected:")
                                        st.write(", ".join(tools))
                                    else:
                                        st.write("No tools selected")
                                elif stage == 'execute_tools':
                                    used = snap.get('tools_used', [])
                                    st.write(f"Tools executed: {len(used)}")
                                    if used:
                                        for t in used:
                                            name = t.get('tool_name','tool') if isinstance(t, dict) else str(t)
                                            st.write(f"- {name}")
                                elif stage == 'reflect':
                                    hyps = snap.get('hypotheses', [])
                                    st.write("Updated hypotheses:")
                                    if hyps:
                                        for i, h in enumerate(hyps[:5], start=1):
                                            st.write(f"{i}. {h.get('diagnosis','')} (confidence: {h.get('confidence',0):.2f})")
                                    else:
                                        st.write("No hypotheses available")
                                elif stage == 'update_memory':
                                    st.write("Clinical memory updated with current case insights")
                                elif stage == 'conclude':
                                    st.write("Finalizing diagnosis and recommendations...")
                                
                                # Raw details (collapsed)
                                with st.expander("Raw stage snapshot"):
                                    st.json(snap)
                        
                        # Show completed stages at bottom (all except current)
                        with completed_stages_container:
                            if len(st.session_state.process_events) > 1:
                                st.subheader("📋 Completed Stages")
                                for idx, ev in enumerate(st.session_state.process_events[:-1], start=1):  # All except latest
                                    stage = ev.get('stage','')
                                    message = ev.get('message','')
                                    with st.expander(f"✅ {idx}. {stage} — {message}", expanded=False):
                                        st.write(f"Stage completed: {stage}")
                                        # Show brief summary of what was done
                                        snap = ev.get('snapshot') or {}
                                        if stage == 'observe':
                                            findings = snap.get('image_findings', {})
                                            if findings:
                                                st.write(f"Found {len(findings)} pathologies")
                                        elif stage == 'retrieve_memory':
                                            cases = snap.get('similar_cases', [])
                                            st.write(f"Retrieved {len(cases)} similar cases")
                                        elif stage == 'think':
                                            hyps = snap.get('hypotheses', [])
                                            st.write(f"Generated {len(hyps)} hypotheses")
                                        elif stage == 'select_tools':
                                            tools = snap.get('selected_tools', [])
                                            st.write(f"Selected {len(tools)} tools")
                                        elif stage == 'execute_tools':
                                            used = snap.get('tools_used', [])
                                            st.write(f"Executed {len(used)} tools")
                                        elif stage == 'reflect':
                                            hyps = snap.get('hypotheses', [])
                                            st.write(f"Refined to {len(hyps)} hypotheses")
                                        elif stage == 'update_memory':
                                            st.write("Memory updated with insights")
                                    # Stage-specific UI
                                    if stage == 'observe':
                                        cols = st.columns([2, 3])
                                        with cols[0]:
                                            try:
                                                st.image(Image.open(temp_image_path), caption="Input X-ray", use_container_width=True)
                                            except Exception:
                                                pass
                                        with cols[1]:
                                            findings = snap.get('image_findings', {})
                                            if findings:
                                                st.write("Detected findings (preview)")
                                                try:
                                                    st.bar_chart({"score": findings})
                                                except Exception:
                                                    st.table({"Pathology": list(findings.keys()), "Score": [f"{v:.3f}" if isinstance(v, (int,float)) else v for v in findings.values()]})
                                        # Heatmaps if available from the event
                                        heatmaps = ev.get('heatmaps')
                                        if heatmaps:
                                            st.write("Heatmaps (Grad-CAM)")
                                            for pathology, hm in list(heatmaps.items())[:4]:
                                                if isinstance(hm, dict) and 'figure' in hm:
                                                    st.write(f"{pathology}")
                                                    st.pyplot(hm['figure'])
                                    elif stage == 'retrieve_memory':
                                        cases = snap.get('similar_cases', [])
                                        st.write(f"Similar cases found: {len(cases)}")
                                        if cases:
                                            preview = [{
                                                "id": c.get('id',''),
                                                "diagnosis": c.get('diagnosis',''),
                                                "similarity": f"{c.get('similarity',0):.2f}"
                                            } for c in cases]
                                            st.table({
                                                "Case ID": [p['id'] for p in preview],
                                                "Diagnosis": [p['diagnosis'] for p in preview],
                                                "Similarity": [p['similarity'] for p in preview]
                                            })
                                    elif stage == 'think':
                                        hyps = snap.get('hypotheses', [])
                                        if hyps:
                                            for i, h in enumerate(hyps[:5], start=1):
                                                st.write(f"{i}. {h.get('diagnosis','')} (confidence: {h.get('confidence',0):.2f})")
                                        else:
                                            st.write("No hypotheses generated yet")
                                    elif stage == 'select_tools':
                                        tools = snap.get('selected_tools', [])
                                        if tools:
                                            st.write("Tools selected:")
                                            st.write(", ".join(tools))
                                        else:
                                            st.write("No tools selected")
                                    elif stage == 'execute_tools':
                                        used = snap.get('tools_used', [])
                                        st.write(f"Tools executed: {len(used)}")
                                        if used:
                                            for t in used:
                                                name = t.get('tool_name','tool') if isinstance(t, dict) else str(t)
                                                st.write(f"- {name}")
                                    elif stage == 'reflect':
                                        hyps = snap.get('hypotheses', [])
                                        st.write("Updated hypotheses:")
                                        if hyps:
                                            for i, h in enumerate(hyps[:5], start=1):
                                                st.write(f"{i}. {h.get('diagnosis','')} (confidence: {h.get('confidence',0):.2f})")
                                        else:
                                            st.write("No hypotheses available")
                                    elif stage == 'update_memory':
                                        st.write("Clinical memory updated with current case insights")
                                    
                                                                            # Raw details (collapsed)
                                        with st.expander("Raw stage snapshot"):
                                            st.json(snap)
                        status_placeholder.info(f"Stage: {event.get('stage','')} ")
                    
                        # Store final results if present
                        if 'final_result' in event:
                            final_results = event['final_result']
                            break
                    
                else:
                    # Legacy, non-streaming execution
                    runner = RadiologyWorkflowRunner()
                    st.info("ℹ️ Using Legacy Radiology Workflow")
                    final_results = runner.run_analysis(temp_image_path, symptoms, generate_heatmaps=generate_heatmaps)

                # Store final results and mark done; render at bottom after stages
                if final_results is not None:
                    st.session_state.last_results = final_results
                    st.session_state.analysis_done = True
                    st.session_state.analysis_in_progress = False
                    # Clean up temporary file
                    if 'temp_image_path' in locals():
                        try:
                            os.unlink(temp_image_path)
                        except Exception:
                            pass

            except Exception as e:
                st.session_state.analysis_in_progress = False
                st.error(f"Analysis failed: {str(e)}")
                if 'temp_image_path' in locals():
                    try:
                        os.unlink(temp_image_path)
                    except Exception:
                        pass
        else:
            if not st.session_state.process_events:
                st.info("Please upload an X-ray image to begin analysis")

        # Always render final report at top when available
        if st.session_state.last_results is not None:
            with report_container:
                st.markdown("---")
                st.header("✅ Final Report")
                display_results(st.session_state.last_results)
                st.markdown("---")
                if st.button("🔄 Analyze another X-ray"):
                    _reset_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>AI Radiology Assistant - Prototype System</p>
        <p>Built with TorchXRayVision, LangChain, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def display_results(results):
    """Display analysis results in the Streamlit interface."""
    
    # Status
    status = results.get('workflow_status', 'unknown')
    
    if status == 'completed':
        st.success("✅ Analysis completed successfully!")
        
        # Triage information
        triage_result = results.get('triage_result', {})
        triage_category = triage_result.get('triage_category', 'UNKNOWN')
        urgency_score = triage_result.get('urgency_score', 0.0)
        
        # Color-coded triage display
        if triage_category == 'EMERGENCY':
            st.error(f"🚨 EMERGENCY")
        elif triage_category == 'URGENT':
            st.warning(f"⚠️ URGENT")
        elif triage_category == 'MODERATE':
            st.info(f"📋 MODERATE")
        else:
            st.success(f"✅ {triage_category}")
        
        # Key findings
        image_analysis = results.get('image_analysis', {})
        key_findings = image_analysis.get('key_findings', {})
        
        if key_findings:
            st.subheader("🔍 Key Findings")
            
            # Create a bar chart of findings
            if len(key_findings) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                pathologies = list(key_findings.keys())
                scores = list(key_findings.values())
                
                colors = ['red' if score > 0.7 else 'orange' if score > 0.5 else 'green' for score in scores]
                
                bars = ax.bar(pathologies, scores, color=colors, alpha=0.7)
                ax.set_ylabel('Confidence Score')
                ax.set_title('Pathology Detection Results')
                ax.set_ylim(0, 1)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display findings in a table
                findings_data = []
                for pathology, score in key_findings.items():
                    severity = "High" if score > 0.7 else "Medium" if score > 0.5 else "Low"
                    findings_data.append([pathology, f"{score:.3f}", severity])
                
                st.table({
                    "Pathology": [row[0] for row in findings_data],
                    "Score": [row[1] for row in findings_data],
                    "Severity": [row[2] for row in findings_data]
                })
        
        # Heatmaps
        heatmaps = image_analysis.get('heatmaps', {})
        if heatmaps:
            st.subheader("🔥 Heatmaps")
            for pathology, heatmap_data in heatmaps.items():
                if 'figure' in heatmap_data:
                    st.write(f"**{pathology}** (Score: {heatmap_data.get('target_score', 0):.3f})")
                    st.pyplot(heatmap_data['figure'])
        
        # Recommendations
        recommendations = triage_result.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # ReAct-specific information
        react_workflow = results.get('react_workflow', {})
        if react_workflow:
            st.subheader("🧠 ReAct Diagnostic Reasoning")
            
            # Iteration and tool information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Iterations", react_workflow.get('iteration_count', 0))
            with col2:
                st.metric("Tools Used", len(react_workflow.get('tools_used', [])))
            with col3:
                st.metric("Reasoning Steps", len(react_workflow.get('reasoning_chain', [])))
            
            # Show hypotheses
            hypotheses = react_workflow.get('hypotheses', [])
            if hypotheses:
                st.write("**🔍 Diagnostic Hypotheses:**")
                for i, hyp in enumerate(hypotheses[:3]):  # Top 3
                    st.write(f"{i+1}. **{hyp['diagnosis']}** (confidence: {hyp['confidence']:.2f})")
                    if hyp.get('evidence'):
                        st.write(f"   Evidence: {', '.join(hyp['evidence'][:3])}")  # Top 3 pieces of evidence
            
            # Show similar cases
            similar_cases = react_workflow.get('similar_cases', [])
            if similar_cases:
                st.write(f"**🧠 Similar Cases Found:** {len(similar_cases)}")
                for i, case in enumerate(similar_cases[:2]):  # Top 2
                    st.write(f"   {i+1}. Case {case.get('id', 'N/A')} - {case.get('diagnosis', 'Unknown')} (similarity: {case.get('similarity', 0):.2f})")
        
        # Diagnostic insights
        diagnostic_insights = results.get('diagnostic_insights', {})
        if diagnostic_insights:
            st.subheader("🎯 Diagnostic Insights")
            
            primary_diagnosis = diagnostic_insights.get('primary_diagnosis')
            if primary_diagnosis:
                st.success(f"**Primary Diagnosis:** {primary_diagnosis}")
            
            differential_diagnoses = diagnostic_insights.get('differential_diagnoses', [])
            if differential_diagnoses:
                st.write("**🔍 Differential Diagnoses:**")
                for i, diff in enumerate(differential_diagnoses[:3]):  # Top 3
                    st.write(f"{i+1}. {diff['diagnosis']} (confidence: {diff['confidence']:.2f})")
        
        # Executive summary
        report = results.get('report', {})
        executive_summary = report.get('executive_summary', '')
        if executive_summary:
            st.subheader("📋 Executive Summary")
            st.write(executive_summary)
        
        # Full report (expandable)
        full_report = report.get('full_report', '')
        if full_report:
            with st.expander("📄 View Full Report"):
                st.text(full_report)
        else:
            # Check if API key is available for report generation
            env_status = check_environment_setup()
            if not env_status["api_key_available"]:
                st.info("""
                **📄 Full Report Not Available**
                
                To enable comprehensive medical report generation, please add your Google AI API key to the `.env` file.
                
                **Steps:**
                1. Copy `env.example` to `.env`
                2. Add your Google AI API key: `GOOGLE_API_KEY=your_key_here`
                3. Restart the application
                
                Basic image analysis results are still available above.
                """)
        

    
    elif status == 'error':
        st.error("❌ Analysis failed")
        errors = results.get('errors', [])
        for error in errors:
            st.error(f"• {error}")
    
    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.write(f"• {warning}")

if __name__ == "__main__":
    main() 