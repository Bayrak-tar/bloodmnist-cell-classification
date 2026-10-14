from PIL import Image
import numpy as np
import os

data = np.load("bloodmnist_224.npz")

splits = ["train", "val", "test"]

base_dir = "bloodmnist_split"

for split in splits:
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"]
    
    for i in range(len(images)):
        label = int(labels[i])
        
        # klasör: train/0, train/1 gibi
        label_dir = os.path.join(base_dir, split, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        img = Image.fromarray(images[i])
        img.save(os.path.join(label_dir, f"{i}.png"))

print("Eğitim / Doğrulama / Test olarak kaydedildi!")