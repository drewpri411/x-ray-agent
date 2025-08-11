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
        
        # Analysis button
        if uploaded_file is not None:
            # Show analysis capabilities
            env_status = check_environment_setup()
            if env_status["api_key_available"]:
                st.success("✅ Full analysis available: Image analysis + LLM report generation")
            else:
                st.info("ℹ️ Basic analysis available: Image analysis only (no LLM report)")
            
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    try:
                        # Initialize workflow runner
                        runner = RadiologyWorkflowRunner()
                        
                        # Run analysis
                        results = runner.run_analysis(temp_image_path, symptoms, generate_heatmaps=generate_heatmaps)
                        
                        # Display results
                        display_results(results)
                        
                        # Save results if requested
                        if save_results and results['workflow_status'] == 'completed':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = f"results_{timestamp}.json"
                            save_json(results, output_file)
                            st.success(f"Results saved to {output_file}")
                        
                        # Clean up temporary file
                        os.unlink(temp_image_path)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        if 'temp_image_path' in locals():
                            os.unlink(temp_image_path)
        else:
            st.info("Please upload an X-ray image to begin analysis")
    
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
            st.error(f"🚨 EMERGENCY - Urgency Score: {urgency_score:.3f}")
        elif triage_category == 'URGENT':
            st.warning(f"⚠️ URGENT - Urgency Score: {urgency_score:.3f}")
        elif triage_category == 'MODERATE':
            st.info(f"📋 MODERATE - Urgency Score: {urgency_score:.3f}")
        else:
            st.success(f"✅ {triage_category} - Urgency Score: {urgency_score:.3f}")
        
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