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

<img width="1919" height="1079" alt="Screenshot 2025-11-21 234955" src="https://github.com/user-attachments/assets/19df3f87-1ec4-4422-b384-42f02a236ed0" />
<img width="1919" height="1013" alt="Screenshot 2025-11-21 231150" src="https://github.com/user-attachments/assets/b7e5d7bc-852f-49a5-a315-362d872a25e5" />
<img width="1912" height="1077" alt="Screenshot 2025-11-21 231209" src="https://github.com/user-attachments/assets/0d2cf8d2-d538-47e5-9cab-7f1113027bfc" />
<img width="388" height="345" alt="Screenshot 2025-11-22 005939" src="https://github.com/user-attachments/assets/f54e0fcb-d300-47e7-8d18-c7e0dd84ed53" />
<img width="790" height="641" alt="Screenshot 2025-11-22 001137" src="https://github.com/user-attachments/assets/f7acbbff-773d-4098-ae2d-8f639c533b67" />
<img width="1038" height="439" alt="Screenshot 2025-11-22 001238" src="https://github.com/user-attachments/assets/8f9d4715-a6a5-4fbc-81ab-bf3c2fad4e23" />
<img width="626" height="292" alt="Screenshot 2025-11-21 204838" src="https://github.com/user-attachments/assets/30bc0d0b-5255-40ba-bbc9-4757f635ce01" />
<img width="515" height="156" alt="Screenshot 2025-11-21 185611" src="https://github.com/user-attachments/assets/4ddd9538-c819-456a-a64e-b076da6193ab" />


## Author
*   MADHANKUMAR S 
Building Agentic workflow Builder with No-Code, AI Agents & Automation | React & N8N Developer | Top Winner – Adya AI Vanij Builder League 2025 | B.Tech Information Technology @ BIT
