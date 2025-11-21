# Chest X-Ray Classification Assessment - Final Report

**Name:** MADHANKUMAR S

In this assessment, I applied deep learning concepts to build a complete end-to-end system for classifying Chest X-Rays (Normal vs. Pneumonia). Below is a summary of what I learned and implemented for each part of the challenge.

---

## Part 1: Understanding & Setup

### What I Learned & Applied
I learned that medical image analysis requires careful data handling to ensure clinical relevance and model robustness.

1.  **Data Preprocessing**:
    *   I understood that real-world X-rays often come in **DICOM** format. I learned that these need to be converted to standard pixel arrays (like PNG/JPG) using windowing techniques to highlight relevant lung structures.
    *   I applied **Normalization** using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to ensure my input data distribution matched the pre-trained model's expectations.
    *   I resized all images to **224x224** pixels to satisfy the input requirements of the CNN architecture.

2.  **Data Augmentation**:
    *   To prevent overfitting, I applied augmentation techniques. I used `RandomHorizontalFlip` because a mirrored X-ray is still anatomically valid, and `RandomRotation` to account for slight misalignments in patient positioning.

3.  **Data Leakage Prevention**:
    *   I learned a critical concept regarding **Patient-Level Split**. I ensured that if a patient had multiple images, all of them went into the same dataset split (Train, Validation, or Test). This prevents the model from "memorizing" a patient's specific anatomy, ensuring the evaluation metrics reflect true generalization.

---

## Part 2: Modelling

### What I Learned & Applied
I focused on building a model that balances accuracy with computational efficiency, suitable for a clinical environment.

1.  **Architecture Choice: ResNet18**:
    *   I chose **ResNet18** because of its residual learning framework, which solves the vanishing gradient problem in deep networks.
    *   I applied **Transfer Learning** by using weights pre-trained on ImageNet. I learned that this allows the model to start with a strong understanding of low-level features (edges, textures, shapes), significantly speeding up training and improving performance on a smaller medical dataset.

2.  **Training Strategy**:
    *   I replaced the final fully connected layer to output just **2 classes** (Normal vs. Pneumonia).
    *   I used the **CrossEntropyLoss** function, which is standard for classification tasks.
    *   I utilized the **Adam optimizer** with a learning rate of `0.001` for faster convergence compared to standard SGD.

3.  **Evaluation & Explainability**:
    *   I validated the model using metrics like **Accuracy**, **Confusion Matrix**, and **Classification Report** (Precision/Recall).
    *   I implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**. I learned that this is crucial for building trust with radiologists, as it visualizes *where* the model is looking (e.g., highlighting lung opacities) rather than just giving a black-box prediction.

---

## Part 3: System Design & Deployment

### What I Learned & Applied
I explored how to take a trained model and integrate it into a real-world hospital workflow.

1.  **PACS Integration**:
    *   I designed a workflow where the AI acts as a "second reader." I proposed using a **DICOM Router** to send X-rays to an inference server, which then sends back a **Secondary Capture (SC)** object containing the prediction and heatmap overlay to the radiologist's workstation.

2.  **Privacy & Compliance**:
    *   I applied **HIPAA/GDPR** principles by ensuring data is de-identified before processing. I proposed an **on-premise deployment** strategy to keep patient data within the hospital's secure firewall, minimizing the risk of data breaches.

3.  **Reliability & Monitoring**:
    *   I learned that AI models can suffer from **Model Drift** (e.g., if a new X-ray machine is installed). I proposed monitoring the distribution of input data and prediction confidence over time.
    *   I also suggested a **"Human-in-the-loop"** feedback mechanism, where radiologists can flag incorrect predictions to help retrain and improve the model continuously.
