# AI Radiology Assistant

An intelligent AI system for chest X-ray analysis, built with TorchXRayVision, LangChain, and LangGraph. This prototype demonstrates automated detection of pulmonary pathologies (pneumonia, pleural effusion, lung opacities) with explainable AI and automated triage capabilities.

## 🚨 Important Disclaimer

**This is a prototype AI system for educational and demonstration purposes only.**
- All findings should be reviewed by qualified healthcare professionals before clinical use
- This system is NOT intended for actual medical diagnosis or treatment decisions
- For research and educational use only

## 🏗️ Architecture Overview

The system uses an agentic architecture with LangGraph orchestration:

```
User Input (Image + Symptoms) 
    ↓
LangGraph Workflow
    ↓
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Symptom Agent   │ Image Agent     │ Triage Agent    │ Report Agent    │
│ (LLM parsing)   │ (TorchXRayVision│ (Urgency assess)│ (LLM generation)│
│                 │ + Grad-CAM)     │                 │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    ↓
Comprehensive Medical Report + Triage Recommendations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google AI API key (for LLM features)
- GPU recommended (for faster inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd x-ray-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env and add your Google AI API key
   ```

4. **Add sample X-ray images:**
   ```bash
   # Place chest X-ray images in the data/ directory
   # See data/README.md for guidelines
   ```

### Usage

#### Command Line Interface

```bash
# Basic analysis
python main.py --image data/sample_xray.jpg --symptoms "Patient has cough and fever for 3 days"

# Save results to file
python main.py --image data/sample_xray.jpg --output results.json

# Verbose logging
python main.py --image data/sample_xray.jpg --verbose
```

#### Web Interface

```bash
# Launch Streamlit web app
streamlit run ui/app_streamlit.py
```

#### Docker

```bash
# Build and run with Docker
docker build -t radiology-assistant .
docker run -p 8501:8501 radiology-assistant
```

## 📁 Project Structure

```
x-ray-agent/
├── agents/                 # LangGraph agent nodes
│   ├── symptom_agent.py   # Symptom parsing with LLM
│   ├── image_agent.py     # X-ray analysis with TorchXRayVision
│   ├── triage_agent.py    # Automated triage assessment
│   └── report_agent.py    # Medical report generation
├── models/                # ML model wrappers
│   ├── image_model.py     # TorchXRayVision classifier
│   └── grad_cam_tool.py   # Explainability with Grad-CAM
├── ui/                    # User interfaces
│   └── app_streamlit.py   # Streamlit web interface
├── utils/                 # Helper utilities
│   └── helpers.py         # Common utility functions
├── data/                  # Sample images and data
├── workflow.py            # LangGraph workflow orchestration
├── main.py               # Command-line entry point
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Required
GOOGLE_API_KEY=your_google_ai_api_key_here

# Optional
GEMINI_MODEL=gemini-1.5-flash
LOG_LEVEL=INFO
TORCHXRAYVISION_MODEL=densenet121-res224-all
GENERATE_HEATMAPS=true
```

### Model Options

Available TorchXRayVision models:
- `densenet121-res224-all` (default) - Trained on multiple datasets
- `densenet121-res224-chex` - CheXpert dataset
- `densenet121-res224-mimic_ch` - MIMIC-CXR dataset

## 🧠 Features

### 1. Automated Pathology Detection
- **Pneumonia** detection with confidence scoring
- **Pleural effusion** identification
- **Lung opacity** analysis
- **Cardiomegaly** detection
- **Pulmonary edema** assessment

### 2. Explainable AI
- **Grad-CAM heatmaps** showing regions of interest
- **Confidence scores** for each pathology
- **Visual explanations** for model decisions

### 3. Intelligent Triage
- **Automated urgency assessment** based on image findings and symptoms
- **Risk stratification** (Emergency, Urgent, Moderate, Routine, Normal)
- **Clinical recommendations** for next steps

### 4. Natural Language Processing
- **Symptom parsing** using LLM
- **Medical report generation** with structured findings
- **Clinical language** output suitable for healthcare providers

### 5. Agentic Workflow
- **LangGraph orchestration** for complex multi-step analysis
- **Error handling** and graceful degradation
- **Modular design** for easy extension

## 📊 Output Format

The system generates comprehensive results including:

```json
{
  "workflow_status": "completed",
  "image_analysis": {
    "key_findings": {
      "Pneumonia": 0.75,
      "Effusion": 0.32
    },
    "heatmaps": {...}
  },
  "triage_result": {
    "triage_category": "URGENT",
    "urgency_score": 0.68,
    "recommendations": [...]
  },
  "report": {
    "executive_summary": "...",
    "full_report": "..."
  }
}
```

## 🔬 Technical Details

### AI Models Used

1. **TorchXRayVision DenseNet-121**
   - Pre-trained on NIH ChestX-ray14, CheXpert, MIMIC-CXR
   - 224x224 image input
   - Multi-label classification for 18+ pathologies

2. **Google Gemini 1.5 Flash**
   - Symptom parsing and medical report generation
   - Structured output for clinical use

3. **Grad-CAM**
   - Explainability for CNN decisions
   - Heatmap generation for transparency

### Performance Considerations

- **GPU acceleration** recommended for real-time analysis
- **Model loading** takes ~30 seconds on first run
- **Inference time** ~2-5 seconds per image (GPU)
- **Memory usage** ~4GB RAM recommended

## 🛠️ Development

### Adding New Agents

1. Create agent class in `agents/` directory
2. Implement required interface methods
3. Add to workflow in `workflow.py`
4. Update state structure if needed

### Extending Pathology Detection

1. Modify `models/image_model.py` for new pathologies
2. Update triage logic in `agents/triage_agent.py`
3. Add new prompts in `agents/report_agent.py`

### Custom Models

Replace TorchXRayVision with custom models:
1. Implement model interface in `models/image_model.py`
2. Update preprocessing in `models/grad_cam_tool.py`
3. Adjust confidence thresholds in triage logic

## 🧪 Testing

### Sample Data

Use publicly available chest X-ray datasets:
- NIH ChestX-ray14 (public domain)
- CheXpert (with attribution)
- MIMIC-CXR (requires approval)

### Validation

```bash
# Test with sample image
python main.py --image data/test_xray.jpg --symptoms "test symptoms"

# Check dependencies
python -c "from utils.helpers import print_dependency_status; print_dependency_status()"
```

## 📚 References

### Research Papers
- [TorchXRayVision: A library for chest X-ray datasets and models](https://arxiv.org/abs/2111.00595)
- [CheXpert: A Large Chest Radiograph Dataset](https://arxiv.org/abs/1901.07031)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

### Datasets
- [NIH ChestX-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- [CheXpert](https://stanfordmlgroup.github.io/projects/chexpert/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)

### API Resources
- [Google AI Studio](https://makersuite.google.com/app/apikey) - Get your Google AI API key
- [Gemini Models](https://ai.google.dev/models/gemini) - Available Gemini models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Regulatory Compliance

### FDA Considerations
- This is a prototype system, not FDA-cleared software
- Clinical deployment would require 510(k) or De Novo clearance
- Follow FDA guidelines for AI/ML medical devices

### HIPAA Compliance
- Use only de-identified data
- Implement appropriate data protection measures
- Follow HIPAA Privacy and Security Rules

### ACR Guidelines
- Ensure explainability and transparency
- Document model limitations and training data
- Provide clear disclaimers about AI assistance

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review error logs
3. Open an issue on GitHub
4. Contact the development team

---

**Remember**: This is a prototype for educational purposes. Always consult qualified healthcare professionals for medical decisions.
