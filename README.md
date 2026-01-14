# 🕳️ Real-Time Pothole Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-orange)
![Hugging Face](https://img.shields.io/badge/Deployment-Hugging%20Face%20Spaces-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project is an **AI-powered Real-Time Pothole Detection System** designed to improve road safety and maintenance. It leverages computer vision techniques using the **YOLOv8** (You Only Look Once) architecture to identify potholes from image and video feeds with high accuracy. The model is deployed as a user-friendly web application using **Gradio**, accessible via browser or mobile for field usage.

## 🏗️ System Architecture

```mermaid
graph LR
    A[Input Source] -->|Webcam/Image| B(Preprocessing)
    B --> C{YOLOv8 Model}
    C -->|Inference| D[Object Detection]
    D -->|Bounding Boxes| E[Post-Processing]
    E --> F[User Interface]
    
    subgraph "Training Pipeline (Google Colab)"
    T1[Roboflow Dataset] --> T2[YOLOv8 Training]
    T2 --> T3[Validation & Metrics]
    T3 --> T4[Export Weights (.pt)]
    end
    
    subgraph "Deployment (Hugging Face Spaces)"
    F[Gradio Web App]
    end
```

## 🚀 Key Features
- **Real-Time Detection**:Capable of processing live camera feeds to detect potholes instantly.
- **High Accuracy**: Trained on a diverse dataset of 600+ labeled pothole images (Roboflow Universe).
- **Lightweight**: Uses `yolov8n` (nano) model for optimized performance on edge devices and CPUs.
- **Web Interface**: Simple, interactive UI built with Gradio for easy testing and demonstration.

## 🛠️ Technology Stack
- **Deep Learning**: Ultralytics YOLOv8
- **Language**: Python 3.10
- **Web Framework**: Gradio
- **Computer Vision**: OpenCV, PIL
- **Training Env**: Google Colab (T4 GPU)
- **Deployment**: Hugging Face Spaces

## 📊 Model Performance
The model was trained for 20 epochs on a T4 GPU.
- **mAP@50**: ~97.% (Precision oriented)
- **Inference Time**: ~20ms per frame (on GPU), <100ms on CPU.

## 📂 Project Structure
```
├── app.py                  # Main Gradio application for inference
├── requirements.txt        # Python dependencies
├── notebooks/
│   ├── Pothole_Detection_Training.ipynb  # Comprehensive Training Pipeline
│   └── colab_training_script.py          # Script version of training
├── best.pt                 # Trained YOLOv8 Model Weights (Upload after training)
└── README.md               # Project Documentation
```

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/mrshibly/Pothole-Detector.git
cd Pothole-Detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
Ensure you have the `best.pt` file in the root directory.
```bash
python app.py
```
Open the local URL provided (usually `http://127.0.0.1:7860`).

## ☁️ Deployment
This app is ready for **Hugging Face Spaces**.
1. Create a new Space (SDK: Gradio).
2. Upload `app.py`, `requirements.txt`, and `best.pt`.
3. The app will build and go live automatically.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License
This project is licensed under the MIT License.
