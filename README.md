# 🤖 Agentic AI Radiology Assistant

An intelligent AI system for chest X-ray analysis, built with **TorchXRayVision**, **LangChain**, and **LangGraph**. This prototype demonstrates **agentic AI** with **ReAct-style reasoning** for automated detection of pulmonary pathologies with explainable AI and automated triage capabilities.

## 🧠 Agentic Architecture Overview

This system implements a **sophisticated agentic AI** that can reason, act, learn, and adapt using the **ReAct pattern** (Reasoning + Acting) orchestrated by **LangGraph**.

### Core Agentic Capabilities:
- **🔍 Autonomous Reasoning**: Multi-step diagnostic reasoning with hypothesis generation and refinement
- **🛠️ Tool Usage**: Dynamic selection and execution of medical diagnostic tools
- **🧠 Memory Integration**: Clinical memory system for learning from past cases
- **🔄 Iterative Refinement**: Continuous improvement through reasoning loops
- **📊 Explainable AI**: Transparent decision-making with evidence tracking

## 🏗️ ReAct + LangGraph Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATION                  │
├─────────────────────────────────────────────────────────────┤
│  START                                                      │
│    ↓                                                        │
│  OBSERVE ──┐                                                │
│    ↓       │                                                │
│  RETRIEVE  │                                                │
│  MEMORY    │                                                │
│    ↓       │                                                │
│  THINK ────┼──► SELECT_TOOLS ──► EXECUTE_TOOLS ──► REFLECT │
│    ↓       │                                                │
│  UPDATE    │                                                │
│  MEMORY    │                                                │
│    ↓       │                                                │
│  OBSERVE ◄─┘                                                │
│    ↓                                                        │
│  CONCLUDE                                                   │
│    ↓                                                        │
│  END                                                        │
└─────────────────────────────────────────────────────────────┘
```

### ReAct Pattern Implementation:

#### 1. **OBSERVE** - Data Collection & Analysis
- X-ray image analysis with TorchXRayVision (DenseNet-121)
- Pathology detection: Pneumonia, Pleural Effusion, Lung Opacities, Cardiomegaly, Edema, Consolidation
- Grad-CAM heatmap generation for explainability
- Patient symptom parsing and categorization

#### 2. **THINK** - Hypothesis Generation
- Generate initial diagnostic hypotheses with confidence scores
- Consider multiple differential diagnoses
- Example: "Cardiogenic Pulmonary Edema (confidence: 0.85)"

#### 3. **ACT** - Tool Usage & External Interactions
- **Dynamic Tool Selection**: Choose relevant diagnostic tools based on findings
- **Tool Execution**:
  - `differential_diagnosis_tool` - Medical knowledge lookup
  - `severity_assessment_tool` - Clinical risk evaluation
  - `clinical_guidelines_tool` - Best practice guidelines
  - `medical_knowledge_lookup_tool` - Literature search
  - `clinical_decision_rules_tool` - Decision support

#### 4. **REFLECT** - Evidence Evaluation
- Evaluate tool outputs against hypotheses
- Refine confidence scores based on evidence
- Update diagnostic reasoning
- Identify knowledge gaps

#### 5. **LOOP** - Iterative Refinement
- Return to OBSERVE with updated context
- Re-analyze with new insights
- Continue until confidence threshold reached (max 3 iterations)

#### 6. **CONCLUDE** - Final Diagnosis
- Present primary diagnosis with confidence
- List differential diagnoses
- Provide clinical recommendations
- Update clinical memory for future cases

## 🛠️ Technical Stack

### Core AI/ML:
- **TorchXRayVision**: Pre-trained DenseNet-121 for chest X-ray analysis
- **PyTorch**: Deep learning framework
- **Grad-CAM**: Explainable AI for heatmap generation

### Agentic Framework:
- **LangGraph**: Workflow orchestration with state management
- **LangChain**: Tool integration and LLM interactions
- **Google Gemini 2.5 Pro**: Large language model for reasoning

### Memory & Learning:
- **Clinical Memory System**: TF-IDF based case similarity
- **Pattern Recognition**: Learning from historical cases
- **Continuous Improvement**: Adaptive reasoning based on outcomes

### User Interface:
- **Streamlit**: Interactive web interface with real-time workflow visualization
- **Live Stage Tracking**: Real-time display of ReAct reasoning process
- **Grad-CAM Visualization**: Interactive heatmap display

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google AI API key (Gemini 2.5 Pro)

### Installation
```bash
git clone <repository>
cd x-ray-agent
pip install -r requirements.txt
```

### Environment Setup
```bash
cp env.example .env
# Add your GEMINI_API_KEY to .env
```

### Run the Application
```bash
# Web Interface
streamlit run ui/app_streamlit.py

# Command Line
python main.py --react path/to/xray.jpg "Patient has cough and fever"
```

## 📊 Agentic Workflow Example

### Input:
- **X-ray Image**: Chest X-ray showing pulmonary findings
- **Symptoms**: "Patient has cough and fever for 3 days"

### ReAct Execution:

#### **Iteration 1:**
1. **OBSERVE**: "Pneumonia: 0.56, Cardiomegaly: 0.66, Edema: 0.57"
2. **THINK**: "Possible CHF with pulmonary edema"
3. **ACT**: Use differential diagnosis tool
4. **REFLECT**: "CHF confirmed, but need to rule out pneumonia"

#### **Iteration 2:**
1. **OBSERVE**: Re-analyze with CHF context
2. **THINK**: "Primary: CHF, Secondary: Pneumonia"
3. **ACT**: Use severity assessment tool
4. **REFLECT**: "Moderate severity, requires prompt evaluation"

#### **CONCLUDE:**
- **Primary Diagnosis**: "Congestive Heart Failure with Pulmonary Edema"
- **Confidence**: 0.85
- **Recommendations**: "Echocardiography, cardiac biomarkers"

## 🧠 Key Agentic Features

### **1. Autonomous Decision Making**
- Agent decides which tools to use based on findings
- Determines when to continue vs. conclude reasoning
- Adapts reasoning strategy based on evidence

### **2. Sophisticated Tool Usage**
- Dynamic tool selection from medical knowledge base
- Structured tool execution with error handling
- Result integration into reasoning process

### **3. Clinical Memory Integration**
- TF-IDF based similarity search for past cases
- Pattern recognition across case history
- Continuous learning and improvement

### **4. Iterative Reasoning**
- Multi-cycle reasoning with evidence accumulation
- Confidence-based stopping criteria
- Hypothesis refinement through tool usage

### **5. Explainable AI**
- Transparent decision process with evidence tracking
- Grad-CAM heatmaps for visual explainability
- Confidence scoring for all diagnoses

## 📁 Project Structure

```
x-ray-agent/
├── agents/                     # Agentic AI components
│   ├── diagnostic_state.py    # ReAct state management
│   ├── react_nodes.py         # ReAct pattern nodes
│   ├── diagnostic_tools.py    # Medical diagnostic tools
│   ├── memory_system.py       # Clinical memory system
│   └── ...
├── workflow_react.py          # LangGraph ReAct workflow
├── workflow.py                # Legacy linear workflow
├── models/                    # ML models
│   ├── image_model.py         # TorchXRayVision wrapper
│   └── grad_cam_tool.py       # Grad-CAM implementation
├── ui/                        # User interface
│   └── app_streamlit.py       # Streamlit with live workflow
└── data/                      # Sample data and memory
```

## 🔬 Advanced Features

### **ReAct Reasoning Engine**
- Multi-iteration diagnostic reasoning
- Tool-based hypothesis validation
- Evidence-driven confidence scoring

### **Clinical Memory System**
- Case similarity using TF-IDF vectors
- Pattern recognition across diagnoses
- Learning from clinical outcomes

### **Real-time Workflow Visualization**
- Live stage-by-stage ReAct process display
- Current stage highlighting
- Completed stages history

### **Medical Tool Integration**
- Differential diagnosis tools
- Severity assessment algorithms
- Clinical guideline lookup
- Medical knowledge base queries

## 🎯 Use Cases

### **Clinical Decision Support**
- Automated chest X-ray interpretation
- Differential diagnosis generation
- Severity assessment and triage
- Clinical guideline recommendations

### **Medical Education**
- Explainable AI for teaching
- Case-based learning
- Pattern recognition training

### **Research & Development**
- Agentic AI for medical applications
- ReAct pattern implementation
- LangGraph workflow orchestration

## ⚠️ Important Notes

### **Regulatory Compliance**
- **SaMD Classification**: Software as a Medical Device
- **HIPAA Considerations**: Patient data privacy
- **ACR Guidelines**: American College of Radiology standards
- **Clinical Validation**: Requires clinical validation for real use

### **Limitations**
- **Prototype Status**: Research and development prototype
- **Clinical Use**: Not approved for clinical decision making
- **Validation**: Requires extensive clinical validation
- **Expert Review**: Always requires radiologist review

## 🤝 Contributing

This is a research prototype demonstrating agentic AI in medical imaging. Contributions are welcome for:
- ReAct pattern improvements
- Additional medical tools
- Enhanced memory systems
- Clinical validation studies

## 📄 License

This project is for research and educational purposes. See LICENSE for details.

---

**Built with ❤️ using LangGraph, ReAct, and modern AI techniques for the future of medical imaging.**
