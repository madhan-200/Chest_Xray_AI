import torch
import torch.nn as nn
from torchvision import models

def generate_model():
    print("Initializing ResNet18 model...")
    # Initialize model with the same architecture as the app
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)  # 2 classes: Normal / Pneumonia
    
    print("Saving model weights to model.pth...")
    # Save the state dictionary
    torch.save(model.state_dict(), "model.pth")
    print("Done! model.pth created successfully.")
    print("WARNING: This model contains random weights and is for TESTING PURPOSES ONLY. It will not provide accurate medical predictions.")

if __name__ == "__main__":
    generate_model()
