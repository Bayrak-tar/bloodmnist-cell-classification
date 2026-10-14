import pathlib

p = pathlib.Path(r"c:\Users\hp\Desktop\Yüksek Lisans\uygulamalı_veri\makale_formatted.md")
c = p.read_text(encoding="utf-8")

replacements = [
    ("ImageNet önceden eğitilmiş ağırlıkları ile transfer öğrenme yaklaşımı uygulanmıştır. DenseNet-121, ResNet-18 ve YOLO26-N modelleri 20 epoch", "DenseNet-121, ResNet-18 ve YOLO26-N modelleri sıfırdan eğitilmiştir. Modeller 20 epoch"),
    ("Tüm modeller ImageNet veri seti üzerinde ön eğitimli ağırlıklar ile başlatılmı�� ve BloodMNIST veri seti üzerinde ince ayar yapılmıştır.", "Tüm modeller BloodMNIST veri seti üzerinde sıfırdan eğitilmiştir."),
    ("Nesne algılama alanından sınıflandırmaya başarılı transfer öğrenme uygulaması gerçekleştirilmiştir.", "Nesne algılama alanından sınıflandırmaya uyarlanmıştır."),
    ("Bu sonuç, transfer öğrenmenin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağladığını doğrulamaktadır", "Bu sonuç, derin öğrenme modellerinin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağlayabildiğini göstermektedir"),
    ("Elde edilen bulgular, transfer öğrenmenin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağladığını göstermektedir.", "Elde edilen bulgular, derin öğrenme modellerinin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağlayabildiğini göstermektedir."),
    ("Transfer öğrenmenin etkinliği, bu yaklaşımın diğer tıbbi görüntüleme alanlarına da uygulanabileceğini göstermektedir.", "Derin öğrenme modellerinin etkinliği, bu yaklaşımın diğer tıbbi görüntüleme alanlarına da uygulanabileceğini göstermektedir."),
]

for old, new in replacements:
    if old in c:
        c = c.replace(old, new)
        print(f"OK: {old[:50]}...")
    else:
        print(f"SKIP: {old[:50]}...")

p.write_text(c, encoding="utf-8")
print("Done!")
