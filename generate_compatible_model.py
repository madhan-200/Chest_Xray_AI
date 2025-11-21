import torch
import torch.nn as nn
from torchvision import models

def generate_compatible_model():
    print("Initializing ResNet18 model (pretrained on ImageNet)...")
    # Use pretrained=True to get valid feature extractors (better than random)
    # This matches the training script reference provided
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer for 2 classes (Normal vs Pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    print("Saving model weights to model.pth...")
    torch.save(model.state_dict(), "model.pth")
    print("Success! model.pth created.")
    print("NOTE: This model has ImageNet weights but the classifier head is untrained.")
    print("It will run in the app but predictions will be random/inaccurate for Pneumonia.")

if __name__ == "__main__":
    generate_compatible_model()
