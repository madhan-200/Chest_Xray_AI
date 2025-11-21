import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)  # 2 classes: Normal / Pneumonia

# Load Trained Weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

classes = ['NORMAL', 'PNEUMONIA']

# Data Transform
test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# Grad-CAM Hooks
gradients = None
activations = None

target_layer = model.layer4[1].conv2

def save_activation(module, input, output):
    global activations
    activations = output

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

def generate_gradcam(img_tensor):
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

    model.zero_grad()
    output[0, class_idx].backward()

    pooled_gradients = torch.mean(gradients, dim=[2,3], keepdim=True)
    cam = torch.relu((pooled_gradients * activations).sum(dim=1)).squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap, class_idx

# Streamlit UI
st.set_page_config(page_title="X-Ray Diagnosis AI", layout="centered")
st.title("Chest X-Ray Classification: Normal vs Pneumonia (v2)")
st.write("By: **MADHANKUMAR S** — Healthcare AI/ML Developer")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    img_tensor = test_transform(image).unsqueeze(0).to(device)

    heatmap, pred_idx = generate_gradcam(img_tensor)
    pred_class = classes[pred_idx]

    with torch.no_grad():
        conf = torch.softmax(model(img_tensor), dim=1)[0][pred_idx].item() * 100

    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: {conf:.2f}%")

    overlay = heatmap * 0.4 + np.array(image.resize((224,224))) * 0.6
    st.image(np.uint8(overlay), caption="Grad-CAM Heatmap", use_column_width=True)

    st.markdown("""
    ### ⚠ Medical Disclaimer  
    The AI model suggests possible Pneumonia.  
    This is NOT a confirmed diagnosis.  
    Please consult a qualified radiologist.
    ---  
    """)
