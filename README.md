#  Forge NLP & Computer Vision 
#  BY SHARAN G S
![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

##  Overview

This repository contains a diverse collection of cutting-edge NLP and CV projects, including:

- **Gesture-based control applications** using computer vision
- **Deep learning models** for image processing and text analysis
- **NLP implementations** including language models, machine translation, and sentiment analysis
- **Computer vision applications** for image restoration, anomaly detection, and quality control
- **AI agents** for real-world applications

---

##  Features

###  Computer Vision Applications

#### Gesture Control Systems
- **Brightness Control** - Control screen brightness with hand gestures
- **Mouse Control** - Navigate your computer using hand tracking
- **Music Player Control** - Control music playback with gestures
- **Presentation Controller** - Navigate slides hands-free
- **Drawing Application** - Create digital art using hand gestures

#### Image Processing
- **Image Restoration** - Advanced image enhancement and restoration
- **Anomaly Detection** - Detect anomalies in images using unsupervised learning
- **Synthetic Image Generation** - Generate synthetic datasets
- **Quality Control** - Bottle quality control using CNN (includes trained model)
- **Manual Filtering** - Custom image filtering implementations

###  Natural Language Processing

#### Language Models
- **Window-based Language Model** - N-gram based language modeling
- **Statistical Machine Translation** - SMT implementation
- **Word2Vec Examples** - Word embedding demonstrations
- **Domain-Adapted LLM** - Low-resource domain adaptation for language models

#### Text Analysis
- **Adaptive Hierarchical RNN ABSA** - Aspect-Based Sentiment Analysis using RNN
- **Syntactic Parser** - Semantic and syntactic parsing

### 🤖 Deep Learning Models

- **CNN Implementations**
  - Padding, stride, and pooling demonstrations
  - Simple CNN demo with visualization
  - Bottle quality control CNN (trained model included)

- **RNN Implementations**
  - Adaptive Hierarchical RNN for ABSA
  - Sequence modeling applications

- **Deep Reinforcement Learning**
  - DRL for image manipulation

###  AI Agents

- **Negotiation Agent** - AI-powered negotiation system with Streamlit UI
  - Simulates marketplace negotiations
  - Multiple seller personas (strict, flexible, desperate)
  - Interactive chat interface

---

##  Project Structure

```
Forge-NLP_CV/
├──  Gesture Control Applications
│   ├── gesture_brightness_control.py
│   ├── gesture_mouse_control.py
│   ├── gesture_music_control.py
│   ├── gesture_presentation_control.py
│   └── gesture_drawing_app.py
│
├──  Computer Vision
│   ├── img-cv.py
│   ├── restoration.py
│   ├── Anomaly-detect.py
│   ├── Synthetic-img.py
│   ├── manual-filter.py
│   ├── cnn_padding_stride_pooling.py
│   ├── simple_cnn_demo.py
│   ├── drl_image_manipulation.py
│   └── unsupervised_anomaly_detection.py
│
├──  Natural Language Processing
│   ├── window_based_language_model.py
│   ├── statistical_machine_translation.py
│   ├── word_to_vector_examples.py
│   ├── llm_low_resource_domain_adaptation.py
│   ├── adaptive_hierarchical_rnn_absa.py
│   ├── syntactic_parser_semanti.py
│   ├── test-nltk.py
│   └── tes-nltk.py
│
├──  AI Agents
│   ├── app.py (Streamlit Negotiation Agent)
│   └── negotiation_agent.py
│
├──  Testing & Utilities
│   ├── test_brightness_control.py
│   └── test_mouse_permissions.py
│
├──  Trained Models
│   ├── adaptive_hierarchical_rnn_absa.pth
│   ├── bottle_quality_control_model.pth
│   └── domain_adapted_llm.pth
│
├──  Outputs & Datasets
│   ├── datasets_synth/
│   ├── outputs/
│   └── training_curves.png
│
└──  Documentation
    ├── README.md
    └── requirements.txt
```

---

##  Installation

### Prerequisites
- Python 3.7 or higher
- Webcam/camera (for gesture control applications)
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sharan-G-S/Forge-NLP-CV-PRJ.git
   cd Forge-NLP-CV-PRJ
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Platform-specific setup**
   
   **macOS:**
   ```bash
   # For brightness control
   brew install brightness
   ```
   
   **Linux:**
   ```bash
   # xrandr is usually pre-installed
   # Grant accessibility permissions for mouse control
   ```
   
   **Windows:**
   ```bash
   # WMI is included with Python on Windows
   # Grant camera and accessibility permissions
   ```

---

##  Usage

### Gesture Control Applications

```bash
# Brightness Control
python gesture_brightness_control.py

# Mouse Control
python gesture_mouse_control.py

# Music Player Control
python gesture_music_control.py

# Presentation Controller
python gesture_presentation_control.py

# Drawing Application
python gesture_drawing_app.py
```

**Controls:**
- Show your hand to the camera
- Use the specified gestures for each application
- Press 'q' to quit

### NLP Applications

```bash
# Language Model
python window_based_language_model.py

# Machine Translation
python statistical_machine_translation.py

# Sentiment Analysis
python adaptive_hierarchical_rnn_absa.py

# Domain Adaptation
python llm_low_resource_domain_adaptation.py
```

### Computer Vision Applications

```bash
# Image Restoration
python restoration.py

# Anomaly Detection
python Anomaly-detect.py

# CNN Demonstrations
python simple_cnn_demo.py
python cnn_padding_stride_pooling.py

# DRL Image Manipulation
python drl_image_manipulation.py
```

### AI Negotiation Agent

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 💻 Technologies Used

### Core Technologies
- **Python 3.7+** - Primary programming language
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision library
- **MediaPipe** - Hand tracking and gesture recognition
- **NumPy** - Numerical computing
- **Streamlit** - Web application framework

### Machine Learning & AI
- **PyTorch** - Neural network implementations
- **NLTK** - Natural language processing
- **scikit-learn** - Machine learning utilities
- **Transformers** - Pre-trained language models

### Computer Vision
- **OpenCV** - Image processing
- **MediaPipe** - Real-time hand tracking
- **PIL/Pillow** - Image manipulation

### Automation & Control
- **PyAutoGUI** - GUI automation for gesture control
- **platform-specific tools** - Brightness and system control

---

##  Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.

**Key Dependencies:**
- `opencv-python >= 4.8.0`
- `mediapipe >= 0.10.0`
- `numpy >= 1.24.0`
- `pyautogui >= 0.9.54`
- `torch` (PyTorch)
- `streamlit`
- `nltk`

---

##  Key Features by Category

### Gesture Control
- ✅ Real-time hand tracking
- ✅ Multiple gesture recognition
- ✅ Smooth cursor movement
- ✅ Visual feedback
- ✅ Cooldown prevention for accidental triggers

### Deep Learning
- ✅ Pre-trained models included
- ✅ Custom CNN architectures
- ✅ RNN for sequence modeling
- ✅ Transfer learning implementations
- ✅ Training visualization

### NLP
- ✅ Multiple language model architectures
- ✅ Sentiment analysis
- ✅ Machine translation
- ✅ Word embeddings
- ✅ Domain adaptation

### Computer Vision
- ✅ Image restoration and enhancement
- ✅ Anomaly detection
- ✅ Quality control systems
- ✅ Synthetic data generation
- ✅ Custom filtering

---

##  Troubleshooting

### Camera Issues
- Ensure camera is not being used by another application
- Grant camera permissions to Terminal/Python
- Check camera index in code (default is 0)

### Gesture Control Issues
- Ensure good lighting conditions
- Keep hand within camera frame
- Grant accessibility permissions for mouse/system control

### Model Loading Issues
- Ensure PyTorch is properly installed
- Check model file paths
- Verify model files are not corrupted

### Permission Issues (macOS)
```bash
# Grant accessibility permissions
System Preferences → Security & Privacy → Privacy → Accessibility
# Add Terminal or your Python IDE
```

---

##  Documentation

Each major component includes inline documentation and comments. For detailed usage:

1. Check the docstrings in individual Python files
2. Review the comments in the code
3. Refer to the training curves and outputs for model performance

---

##  Learning Resources

This repository demonstrates:
- **Computer Vision**: Hand tracking, gesture recognition, image processing
- **Deep Learning**: CNN, RNN, transfer learning, reinforcement learning
- **NLP**: Language models, sentiment analysis, machine translation
- **AI Agents**: Negotiation systems, decision-making algorithms
- **Software Engineering**: Clean code, modular design, testing

---

##  Contributing

This is a personal portfolio project. However, suggestions and feedback are welcome!

---

##  License

This project is open source and available under the MIT License.

---

##  Author
**Sharan G S**
---



---

<div align="center">

### Made with 💚 from Sharan G S


</div>
