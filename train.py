import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ==========================================
# PART 1: Data Preprocessing & Setup
# ==========================================

# Configuration
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Data Directory Logic
# 1. Check local directory (created by setup_dummy_data.py or user provided)
# 2. Check Kaggle directory (original default)
if os.path.exists("chest_xray"):
    DATA_DIR = "chest_xray"
    print(f"Found local data directory: {DATA_DIR}")
elif os.path.exists("/kaggle/input/chest-xray-pneumonia/chest_xray"):
    DATA_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray"
    print(f"Found Kaggle data directory: {DATA_DIR}")
else:
    DATA_DIR = "chest_xray" # Default fallback
    print(f"WARNING: Data directory not found. Defaulting to '{DATA_DIR}'")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transforms
# We use 224x224 for ResNet compatibility.
# Normalization values are standard for ImageNet pretrained models.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(), # Augmentation to reduce overfitting
    transforms.RandomRotation(10),     # Slight rotation for robustness
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_data_loaders(data_dir):
    # Check if directory exists to avoid crashing if data isn't there
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory '{data_dir}' not found. Please update DATA_DIR.")
        return None, None, None, None

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_data.classes

# ==========================================
# PART 2: Model Architecture (Transfer Learning)
# ==========================================

def build_model(num_classes=2):
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers to use them as feature extractors
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    # ResNet18's fc layer has 512 input features
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model.to(device)

# ==========================================
# PART 3: Training Loop
# ==========================================

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    train_losses = []
    train_accs = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")
        
    return train_losses, train_accs

# ==========================================
# PART 4: Evaluation
# ==========================================

def evaluate_model(model, test_loader, classes):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    # Classification Report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))
    
    return cm

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Setup Data
    train_loader, val_loader, test_loader, classes = get_data_loaders(DATA_DIR)
    
    if train_loader:
        # 2. Build Model
        model = build_model(len(classes))
        
        # 3. Setup Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # Optimize only the classifier parameters
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        
        # 4. Train
        train_losses, train_accs = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
        
        # 5. Evaluate
        evaluate_model(model, test_loader, classes)
        
        # 6. Save Model
        torch.save(model.state_dict(), "model.pth")
        print("\nModel saved to model.pth")
    else:
        print("Skipping training as data was not found.")
