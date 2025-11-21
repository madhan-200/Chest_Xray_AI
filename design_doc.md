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
