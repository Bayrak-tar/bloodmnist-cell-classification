import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ========================
# Ayarlar
# ========================
DATA_DIR = "bloodmnist_split"
NUM_CLASSES = 8
MODEL_PATH = os.path.join("resnet-18", "checkpoints", "best_model.pth")

SINIF_ADLARI = [
    "Bazofil",                  # 0
    "Eozinofil",                # 1
    "Eritroblast",              # 2
    "Olgunlaşmamış Granülosit", # 3
    "Lenfosit",                 # 4
    "Monosit",                  # 5
    "Nötrofil",                 # 6
    "Trombosit",                # 7
]

# ========================
# Görüntü Dönüşümleri
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    print(f"Model: {MODEL_PATH}")

    # ========================
    # Model Yükleme
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model yüklendi.\n")

    # ========================
    # Test seti üzerinde tahmin
    # ========================
    test_dir = os.path.join(DATA_DIR, "test")
    class_dirs = sorted(os.listdir(test_dir))

    all_preds = []
    all_labels = []

    for label_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        img_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Sınıf {class_name} ({SINIF_ADLARI[label_idx]}): {len(img_files)} görüntü değerlendiriliyor...")

        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                all_preds.append(pred.item())
                all_labels.append(label_idx)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ========================
    # Metrikler
    # ========================
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest: {len(all_labels)} görüntü")
    print(f"Test Doğruluğu: {accuracy * 100:.2f}%\n")

    print("=" * 60)
    print("Sınıflandırma Raporu")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=SINIF_ADLARI)
    print(report)

    # Raporu dosyaya kaydet
    with open(os.path.join("resnet-18", "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Doğruluğu: {accuracy * 100:.2f}%\n\n")
        f.write("Sınıflandırma Raporu\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print("Rapor 'resnet-18/classification_report.txt' olarak kaydedildi.")

    # ========================
    # Confusion Matrix
    # ========================
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Karışıklık Matrisi - ResNet-18", fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(SINIF_ADLARI, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(SINIF_ADLARI, fontsize=9)

    # Hücrelere sayı yaz
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("resnet-18", "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Karışıklık matrisi 'resnet-18/confusion_matrix.png' olarak kaydedildi.")
