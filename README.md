# Chest X-Ray Classification AI

## Overview
This project implements a deep learning solution for classifying Chest X-Ray images as **Normal** or **Pneumonia**. It utilizes a **ResNet18** architecture with transfer learning to achieve high accuracy and includes **Grad-CAM** (Gradient-weighted Class Activation Mapping) for model explainability, highlighting the regions in the X-ray that influenced the prediction.

The project also includes a comprehensive system design for integrating this AI model into a hospital's **PACS** (Picture Archiving and Communication System) workflow, ensuring data privacy (HIPAA/GDPR compliance) and model monitoring.

## Features
*   **Deep Learning Model**: ResNet18 trained on Chest X-Ray data.
*   **Explainability**: Grad-CAM heatmaps to visualize model focus areas.
*   **User Interface**: Interactive web application built with **Streamlit**.
*   **Real-world Integration**: Detailed design for DICOM handling and PACS integration.
*   **Privacy Focused**: Strategies for de-identification and on-premise deployment.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Interact with the App:**
    *   Open the provided local URL in your browser.
    *   Upload a Chest X-Ray image (JPG, PNG).
    *   View the prediction (Normal/Pneumonia) and the confidence score.
    *   Examine the Grad-CAM heatmap overlay to understand *why* the model made its prediction.

## Folder Structure

```
├── chest_xray/               # Dataset directory (Train/Val/Test)
├── assessment_report.md      # Project assessment report
├── design_doc.md             # Technical report & System Design (PACS, Privacy)
├── generate_compatible_model.py # Script to generate/convert model weights
├── generate_model.py         # Script to generate initial model
├── model.pth                 # Trained PyTorch model weights
├── packages.txt              # System-level packages
├── requirements.txt          # Python dependencies
├── setup_dummy_data.py       # Script to setup dummy data for testing
├── streamlit_app.py          # Main Streamlit web application
└── train.py                  # Model training script
```

## Dataset Info
The model is trained on a dataset of Chest X-Ray images (Anterior-Posterior).
*   **Preprocessing**: Images are resized to **224x224** and normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
*   **Augmentation**: Random horizontal flips and rotations are applied during training to improve generalization.
*   **Splitting Strategy**: Patient-level splitting is recommended to prevent data leakage.

## Results
*   **Model**: ResNet18 (Pre-trained on ImageNet, fine-tuned).
*   **Classes**: NORMAL, PNEUMONIA.
*   **Output**: Prediction label, Confidence score, and Grad-CAM visualization.

## Author
*   MADHANKUMAR S 
Building Agentic workflow Builder with No-Code, AI Agents & Automation | React & N8N Developer | Top Winner – Adya AI Vanij Builder League 2025 | B.Tech Information Technology @ BIT
