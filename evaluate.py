import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ========================
# Ayarlar
# ========================
DATA_DIR = "bloodmnist_split"
NUM_CLASSES = 8
MODEL_PATH = os.path.join("checkpoints", "best_model.pt")

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

if __name__ == '__main__':
    print(f"Model: {MODEL_PATH}")

    # ========================
    # Model Yükleme
    # ========================
    model = YOLO(MODEL_PATH)
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
        results = model.predict(source=class_path, imgsz=224, verbose=False)
        for r in results:
            pred = int(r.probs.top1)
            all_preds.append(pred)
            all_labels.append(label_idx)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ========================
    # Metrikler
    # ========================
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test: {len(all_labels)} görüntü")
    print(f"Test Doğruluğu: {accuracy * 100:.2f}%\n")

    print("=" * 60)
    print("Sınıflandırma Raporu")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=SINIF_ADLARI)
    print(report)

    # Raporu dosyaya kaydet
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Doğruluğu: {accuracy * 100:.2f}%\n\n")
        f.write("Sınıflandırma Raporu\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print("Rapor 'classification_report.txt' olarak kaydedildi.")

    # ========================
    # Confusion Matrix
    # ========================
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Karışıklık Matrisi", fontsize=14)
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
    plt.savefig("karisiklik_matrisi.png", dpi=150)
    plt.show()
    print("Karışıklık matrisi 'karisiklik_matrisi.png' olarak kaydedildi.")
