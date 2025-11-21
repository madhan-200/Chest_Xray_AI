import os
import numpy as np
from PIL import Image

# Configuration
DATA_DIR = "chest_xray"
SETS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)
NUM_IMAGES = 5  # Images per class per set

def create_dummy_data():
    print(f"Creating dummy dataset in '{DATA_DIR}'...")
    
    if os.path.exists(DATA_DIR):
        print(f"Directory '{DATA_DIR}' already exists. Skipping creation.")
        return

    for dataset in SETS:
        for class_name in CLASSES:
            # Create directory: chest_xray/train/NORMAL
            dir_path = os.path.join(DATA_DIR, dataset, class_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create dummy images
            for i in range(NUM_IMAGES):
                # Random noise image
                img_array = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                # Save
                img.save(os.path.join(dir_path, f"img_{i}.jpg"))
    
    print("Dummy dataset created successfully!")
    print(f"Structure: {DATA_DIR}/[train,val,test]/[NORMAL,PNEUMONIA]")

if __name__ == "__main__":
    create_dummy_data()
