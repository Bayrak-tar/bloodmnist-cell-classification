import os
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ========================
# Sınıf İsimleri ve Açıklamaları
# ========================
SINIF_ADLARI = [
    "Basophil",                  # 0
    "Eosinophil",                # 1
    "Erythroblast",              # 2
    "Immature Granulocyte",      # 3
    "Lymphocyte",                # 4
    "Monocyte",                  # 5
    "Neutrophil",               # 6
    "Platelet",                  # 7
]

SINIF_ACIKLAMALARI = {
    0: "Basophil - En az sayıda bulunan beyaz kan hücresi. Parazitik enfeksiyonlar ve alerjik reaksiyonlarda rol oynar.",
    1: "Eosinophil - Parazitik enfeksiyonlara ve alerjik tepkilere karşı savunmada görevli beyaz kan hücresi.",
    2: "Erythroblast - Kemik iliğinde olgunlaşmamış kırmızı kan hücresi (eritrosit) öncüsü.",
    3: "Immature Granulocyte - Nötrofil, eozinofil ve bazofillerin olgunlaşmamış formlarını kapsayan grup.",
    4: "Lymphocyte - Bağışıklık sisteminin adaptif yanıtından sorumlu beyaz kan hücresi (T ve B hücreleri).",
    5: "Monocyte - Dokulara geçerek makrofaja dönüşebilen, fagositoz yapan en büyük beyaz kan hücresi.",
    6: "Neutrophil - En yaygın beyaz kan hücresi. Bakteriyel enfeksiyonlara karşı ilk savunma hattı.",
    7: "Platelet - Kan pıhtılaşması ve yara iyileşmesinde görevli küçük hücre parçacığı.",
}

# ========================
# Model Yolları
# ========================
MODEL_PATHS = {
    "ResNet-18": os.path.join("resnet-18", "checkpoints", "best_model.pth"),
    "DenseNet-121": os.path.join("densenet121", "checkpoints", "best_model.pth"),
    "YOLO26-N": os.path.join("yolo26", "checkpoints", "best_model.pt"),
}

# ========================
# PyTorch Modelleri İçin Dönüşümler
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========================
# Model Yükleme Fonksiyonları
# ========================
def load_resnet18(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def load_densenet121(path):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 8)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def load_yolo26(path):
    from ultralytics import YOLO
    return YOLO(path)

# ========================
# Tahmin Fonksiyonu
# ========================
def predict(image, model_name):
    if image is None:
        return "Lütfen bir görüntü yükleyin.", None

    img = Image.fromarray(image).convert("RGB")

    if model_name == "YOLO26-N":
        from ultralytics import YOLO
        model = load_yolo26(MODEL_PATHS[model_name])
        results = model.predict(source=img, imgsz=224, verbose=False)
        r = results[0]
        top1_idx = int(r.probs.top1)
        probs = r.probs.data.cpu().numpy()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "ResNet-18":
            model = load_resnet18(MODEL_PATHS[model_name])
        elif model_name == "DenseNet-121":
            model = load_densenet121(MODEL_PATHS[model_name])
        else:
            return "Bilinmeyen model.", None

        model.to(device)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            # Sıcaklık ölçeklendirme: >1 değerler aşırı güveni azaltır
            temperature = 2.0
            scaled_outputs = outputs / temperature
            probs = torch.softmax(scaled_outputs, dim=1).cpu().numpy()[0]
            top1_idx = int(np.argmax(probs))

    # Sonuç metni
    sinif_adi = SINIF_ADLARI[top1_idx]
    aciklama = SINIF_ACIKLAMALARI[top1_idx]
    guven = probs[top1_idx] * 100

    sonuc = (
        f"**Tahmin Edilen Sınıf:** {sinif_adi}\n"
        f"**Sınıf Index:** {top1_idx}\n"
        f"**Güven Skoru:** %{guven:.2f}\n\n"
        f"**Açıklama:** {aciklama}"
    )

    # Güven skorları tablosu (yüksekten düşüğe sıralı)
    skorlar = []
    for i in range(8):
        skorlar.append([SINIF_ADLARI[i], f"%{probs[i]*100:.2f}", probs[i]])
    skorlar.sort(key=lambda x: x[2], reverse=True)
    # Sıra numarası olmadan sadece sınıf adı ve güven oranı
    guven_tablosu = [[s[0], s[1]] for s in skorlar]

    return sonuc, guven_tablosu


# ========================
# Gradio Arayüzü
# ========================
with gr.Blocks(
    title="BloodMNIST Sınıflandırma",
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="pink"),
    css="""
        .main-title {text-align: center; margin-bottom: 5px;}
        .subtitle {text-align: center; color: #666; margin-bottom: 20px;}
        .footer {text-align: center; margin-top: 20px; color: #999; font-size: 12px;}
    """
) as demo:

    gr.Markdown(
        """
        # 🩸 BloodMNIST Kan Hücresi Sınıflandırma
        ### Kan hücresi görüntüsü yükleyerek yapay zeka modeli ile sınıflandırma yapın
        """,
        elem_classes=["main-title"]
    )

    with gr.Row():
        # Sol panel - Giriş
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Giriş")
            image_input = gr.Image(
                type="numpy",
                label="Kan Hücresi Görüntüsü Yükleyin",
                height=300,
            )
            model_dropdown = gr.Dropdown(
                choices=["ResNet-18", "DenseNet-121", "YOLO26-N"],
                value="ResNet-18",
                label="Model Seçin",
                info="Sınıflandırma için kullanılacak modeli seçin",
            )
            predict_btn = gr.Button(
                "🔍 Sınıflandır",
                variant="primary",
                size="lg",
            )

            gr.Markdown(
                """
                ---
                **Model Bilgileri:**
                - **ResNet-18:** 18 katmanlı residual ağ
                - **DenseNet-121:** 121 katmanlı dense bağlantılı ağ
                - **YOLO26-N:** YOLO26 nano sınıflandırma modeli
                """
            )

        # Sağ panel - Sonuçlar
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Sonuçlar")
            result_text = gr.Markdown(
                value="*Sonuçlar burada görünecek...*",
                label="Tahmin Sonucu",
            )
            prob_plot = gr.Dataframe(
                label="Güven Skorları Tablosu",
                headers=["Sınıf Adı", "Güven Oranı"],
                interactive=False,
                wrap=True,
                column_widths=["50%", "50%"],
                datatype=["str", "str"],
            )

    # Sınıf referans tablosu
    gr.Markdown("### 📋 Sınıf Referans Tablosu")
    ref_data = [[i, SINIF_ADLARI[i], SINIF_ACIKLAMALARI[i]] for i in range(8)]
    gr.Dataframe(
        value=ref_data,
        headers=["Index", "Sınıf Adı", "Açıklama"],
        interactive=False,
        wrap=True,
    )

    gr.Markdown(
        "BloodMNIST Veri Seti - Kan Hücresi Sınıflandırma | Uygulamalı Veri Analizi",
        elem_classes=["footer"]
    )

    # Buton tıklama olayı
    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_text, prob_plot],
    )

    # Görüntü yüklendiğinde otomatik tahmin
    image_input.change(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_text, prob_plot],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

