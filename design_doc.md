# Chest X-Ray Classification - Technical Report

## Part 1: Understanding & Setup

### 1. Data Exploration & Preprocessing Strategy
*   **DICOM Handling**: In a real clinical setting, X-rays come in DICOM format. I would use `pydicom` to read these files, extract the pixel array, and apply windowing/leveling (VOI LUT) to convert them to standard 8-bit grayscale images (PNG/JPG) for the model.
*   **Normalization**: I use ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) because I am using a pre-trained ResNet. This ensures the input distribution matches what the model expects.
*   **Resizing**: Images are resized to **224x224** to match the standard input size of ResNet18.
*   **Augmentation**: To prevent overfitting, I apply:
    *   `RandomHorizontalFlip`: Mirrors the X-ray (common variance).
    *   `RandomRotation(10)`: Simulates slight patient misalignment.
*   **Data Split & Leakage**:
    *   **Strategy**: 70% Train, 15% Validation, 15% Test.
    *   **Patient Leakage**: Crucially, the split must be done at the **Patient ID** level, not the Image ID level. If a patient has multiple X-rays, all of them must go into the *same* set (e.g., all in Train). If we split by image, the model might "memorize" a patient's specific bone structure in Train and recognize it in Test, leading to artificially high accuracy (Data Leakage).

## Part 2: Modelling Choices

### 1. Architecture: ResNet18 (Transfer Learning)
*   **Why ResNet18?**:
    *   **Efficiency**: It is lightweight and fast, making it suitable for deployment on hospital servers with limited resources or even edge devices.
    *   **Performance**: Residual connections prevent the vanishing gradient problem, allowing for effective deep feature extraction.
    *   **Transfer Learning**: By initializing with **ImageNet weights**, the model starts with a robust understanding of edges, textures, and shapes. We only need to fine-tune it to recognize the specific patterns of Pneumonia (consolidations, opacities).
*   **Hyperparameters**:
    *   **Batch Size**: 16 (Safe for standard GPUs).
    *   **Learning Rate**: 0.001 (Standard starting point for Adam optimizer).
    *   **Optimizer**: Adam (Adaptive learning rates help converge faster than SGD).
    *   **Loss Function**: CrossEntropyLoss (Standard for multi-class classification).

---

# Part 3: Discussion / System Design

## 1. Integration into Hospital PACS (Picture Archiving and Communication System)

To integrate this AI model into an existing hospital PACS workflow, I would design a **DICOM Router/Gateway** solution.

### Workflow:
1.  **Image Acquisition**: X-ray is taken and sent to the PACS server.
2.  **Auto-Routing**: A DICOM router (e.g., Orthanc or dcm4chee) is configured to forward a copy of Chest X-ray studies to the AI Inference Server.
3.  **Preprocessing & Inference**:
    *   The AI server converts DICOM to pixel arrays (handling windowing/leveling).
    *   Runs the `model.pth` inference.
    *   Generates a **DICOM Secondary Capture (SC)** or **Structured Report (SR)** containing the prediction and the Grad-CAM heatmap.
4.  **Result Push**: The AI server sends this new DICOM object back to the PACS.
5.  **Radiologist Review**: When the radiologist opens the study, they see the original X-ray along with the AI's "opinion" as an additional series or overlay.

### Key Components:
*   **Inference Engine**: Dockerized container running the PyTorch model (FastAPI/Flask wrapper).
*   **DICOM Listener/Sender**: Pydicom or DCMTK for handling DICOM network protocols (C-STORE).
*   **Queue System**: RabbitMQ or Redis to handle high throughput of incoming images.

---

## 2. Data Privacy and Compliance (HIPAA / GDPR)

Deploying in a healthcare setting requires strict adherence to privacy laws.

*   **De-identification**: The AI model does not need Patient Name, ID, or DOB to make a prediction. All PHI (Protected Health Information) should be stripped or anonymized *before* the image data enters the inference pipeline if the server is external.
*   **On-Premise Deployment**: To minimize risk, the AI server should be deployed **on-premise** (inside the hospital's firewall). This avoids sending patient data to the cloud.
*   **Encryption**: All data in transit (DICOM TLS) and at rest must be encrypted.
*   **Audit Logs**: Every access to the image and every prediction generated must be logged (Who, When, What) for compliance auditing.

---

## 3. Monitoring Model Drift and Reliability

AI models can degrade over time if the input data changes (e.g., new X-ray machine, different protocol).

### Monitoring Strategy:
*   **Data Drift**: Monitor statistical distribution of input image intensities and metadata (e.g., Manufacturer tag). If the distribution shifts significantly from the training set, trigger an alert.
*   **Prediction Drift**: Track the ratio of "Normal" vs "Pneumonia" predictions over time. A sudden spike in Pneumonia cases might indicate a model error (or a pandemic).
*   **Radiologist Feedback Loop**: Implement a simple UI mechanism for radiologists to "Agree" or "Disagree" with the AI.
    *   *Disagree* cases are flagged as "Hard Negatives" and sent for re-training (Active Learning).
*   **Clinical Calibration**: Periodically test the model against a "Golden Set" of confirmed cases to ensure Sensitivity/Specificity remains within safety thresholds.

### Reliability:
*   **Confidence Thresholds**: Do not show predictions if confidence is low (e.g., < 70%). Mark as "Indeterminate".
*   **Fail-Safe**: The system must fail silently. If the AI server goes down, it should not block the radiologist from viewing the original images.
