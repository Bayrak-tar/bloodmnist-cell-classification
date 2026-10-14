import os
import shutil
import csv
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ========================
# Ayarlar
# ========================
DATA_DIR = "bloodmnist_split"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224
CHECKPOINT_DIR = "checkpoints"

if __name__ == '__main__':
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========================
    # Model (YOLO26-N Classification)
    # ========================
    model = YOLO("yolo26n-cls.pt")

    print(f"Model: YOLO26-N Sınıflandırma")
    print(f"Veri seti: {DATA_DIR}\n")

    # ========================
    # Eğitim
    # ========================
    results = model.train(
        data=DATA_DIR,
        epochs=NUM_EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE,
        workers=2,
        project="runs",
        name="yolo26_cls",
        exist_ok=True,
    )

    # ========================
    # En iyi modeli checkpoints klasörüne kopyala
    # ========================
    run_dir = os.path.join("runs", "classify", "runs", "yolo26_cls")
    best_src = os.path.join(run_dir, "weights", "best.pt")
    last_src = os.path.join(run_dir, "weights", "last.pt")

    if os.path.exists(best_src):
        shutil.copy(best_src, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        print(f"En iyi model kaydedildi: {CHECKPOINT_DIR}/best_model.pt")
    if os.path.exists(last_src):
        shutil.copy(last_src, os.path.join(CHECKPOINT_DIR, "last_model.pt"))

    # ========================
    # Eğitim Sonuçlarını Dosyaya Kaydet
    # ========================
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Sütun isimlerindeki boşlukları temizle
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]

        with open("training_results.txt", "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("BloodMNIST - Eğitim Sonuçları (YOLO26)\n")
            f.write("=" * 60 + "\n\n")

            f.write("--- Ayarlar ---\n")
            f.write(f"Model         : YOLO26-N (Classification, Transfer Learning)\n")
            f.write(f"Epoch         : {NUM_EPOCHS}\n")
            f.write(f"Batch Size    : {BATCH_SIZE}\n")
            f.write(f"Learning Rate : {LEARNING_RATE}\n")
            f.write(f"Image Size    : {IMG_SIZE}\n\n")

            f.write("--- Epoch Bazlı Sonuçlar ---\n")
            f.write(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Top-1 Acc':>10}\n")
            f.write("-" * 50 + "\n")

            for row in rows:
                epoch = row.get("epoch", "")
                train_loss = row.get("train/loss", "0")
                val_loss = row.get("val/loss", "0")
                top1 = row.get("metrics/accuracy_top1", "0")
                f.write(f"{int(float(epoch))+1:>5} | {float(train_loss):>10.4f} | {float(val_loss):>10.4f} | {float(top1)*100:>9.2f}%\n")

            best_top1 = max(float(r.get("metrics/accuracy_top1", "0")) for r in rows)
            f.write(f"\n--- Özet ---\n")
            f.write(f"En İyi Top-1 Accuracy: {best_top1*100:.2f}%\n")

        print("Eğitim sonuçları 'training_results.txt' olarak kaydedildi.")

    # ========================
    # Eğitim Grafikleri
    # ========================
    if os.path.exists(results_csv):
        epochs = [int(float(r["epoch"]))+1 for r in rows]
        train_losses = [float(r["train/loss"]) for r in rows]
        val_losses = [float(r["val/loss"]) for r in rows]
        top1_accs = [float(r["metrics/accuracy_top1"])*100 for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, train_losses, label="Eğitim Kaybı")
        ax1.plot(epochs, val_losses, label="Doğrulama Kaybı")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Kayıp")
        ax1.set_title("Kayıp Grafiği")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, top1_accs, label="Doğruluk")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_title("Doğruluk Grafiği")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        plt.show()
        print("Eğitim grafikleri 'training_curves.png' olarak kaydedildi.")

    print(f"\nEğitim tamamlandı! Detaylı sonuçlar: {run_dir}/")
