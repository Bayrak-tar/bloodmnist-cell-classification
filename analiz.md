# BloodMNIST Veri Seti - Model Karşılaştırma Analizi

## 1. Genel Bakış

Bu çalışmada BloodMNIST veri seti üzerinde 3 farklı derin öğrenme modeli transfer öğrenme yaklaşımı ile eğitilmiş ve performansları karşılaştırılmıştır.

| Özellik | DenseNet-121 | ResNet-18 | YOLO26-N |
|---------|-------------|-----------|----------|
| Yaklaşım | Transfer Learning | Transfer Learning | Transfer Learning (Classification) |
| Epoch | 20 | 20 | 20 |
| Batch Size | 32 | 32 | 32 |
| Learning Rate | 0.0001 | 0.0001 | 0.0001 |
| Optimizer | Adam | Adam | - |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | - |
| Loss | CrossEntropyLoss (weighted) | CrossEntropyLoss (weighted) | - |
| Görüntü Boyutu | - | - | 224 |

**Veri Seti Dağılımı:**
- Eğitim: 11.959 görüntü
- Doğrulama (Validation): 1.712 görüntü
- Test: 3.421 görüntü
- Sınıf Sayısı: 8 (Bazofil, Eozinofil, Eritroblast, Olgunlaşmamış Granülosit, Lenfosit, Monosit, Nötrofil, Trombosit)

---

## 2. Eğitim Performans Özeti

| Metrik | DenseNet-121 | ResNet-18 | YOLO26-N |
|--------|-------------|-----------|----------|
| En İyi Val Doğruluğu | **99.42%** | 99.36% | 98.95% |
| Test Doğruluğu | 99.09% | 98.89% | **99.12%** |
| Son Eğitim Kaybı | **0.0023** | 0.0044 | 0.0327 |
| Son Doğrulama Kaybı | **0.0208** | 0.0263 | 0.0273 |
| Son Eğitim Doğruluğu | **99.91%** | 99.80% | - |

---

## 3. Model Bazlı Detaylı Analiz

### 3.1 DenseNet-121

**Eğitim Süreci:**
- İlk epoch'tan itibaren yüksek performans göstermiştir (Epoch 1: Val Doğruluğu %98.66).
- Eğitim süreci boyunca **kararlı ve monoton** bir iyileşme gözlemlenmiştir.
- Eğitim kaybı sürekli azalarak 0.0023'e, doğrulama kaybı ise 0.0208'e inmiştir.
- Epoch 9'dan itibaren öğrenme hızı planlayıcı devreye girmiş ve daha ince ayar yapılmıştır.
- Overfitting belirtisi minimumdur: eğitim-doğrulama doğruluğu farkı sadece ~%0.5'tir.

**Güçlü Yönler:**
- En yüksek doğrulama doğruluğu (%99.42)
- En düşük eğitim ve doğrulama kaybı değerleri
- En kararlı eğitim süreci (kayıp grafiğinde dalgalanma yok)
- Sınıf bazında en dengeli performans

**Zayıf Yönler:**
- Daha fazla parametre nedeniyle diğer modellere kıyasla daha ağır bir mimari

### 3.2 ResNet-18

**Eğitim Süreci:**
- İlk epoch'larda hızlı öğrenme gerçekleşmiştir (Epoch 1: Val Doğruluğu %97.14).
- Epoch 4 ve 6'da doğrulama kaybında belirgin **sıçramalar** yaşanmıştır (0.0966 ve 0.0889).
- Bu durum eğitim sürecinde **kararsızlık** olduğunu göstermektedir.
- Epoch 11'den sonra daha kararlı bir eğitim süreci gözlemlenmiştir.
- Son epoch'larda eğitim doğruluğu %99.80'e, doğrulama doğruluğu %99.24'e ulaşmıştır.

**Güçlü Yönler:**
- Hafif mimari, hızlı çalışma süresi
- Sonuç olarak yüksek doğruluk oranı (%99.36 doğrulama, %98.89 test)
- Transfer learning ile etkili adaptasyon

**Zayıf Yönler:**
- Eğitim sürecinde belirgin dalgalanmalar (Epoch 4 ve 6)
- DenseNet-121'e kıyasla daha yüksek doğrulama kaybı (0.0263 vs 0.0208)
- Üç modelden en düşük test doğruluğu (%98.89)

### 3.3 YOLO26-N

**Eğitim Süreci:**
- İlk epoch'ta en yüksek eğitim kaybına sahiptir (0.7518) - bu daha yavaş bir başlangıca işaret eder.
- Epoch 8'den itibaren %98+ doğruluk seviyesine ulaşmıştır.
- Son epoch'larda eğitim kaybı 0.0327'ye kadar düşmesine rağmen DenseNet ve ResNet'e kıyasla daha yüksek kalmıştır.
- Doğrulama kaybı 0.0273 ile makul bir seviyede sonlanmıştır.

**Güçlü Yönler:**
- **En yüksek test doğruluğu (%99.12)** - genelleme kapasitesi en iyi model
- Doğrulama ve test arasındaki fark en az olan model (iyi genelleme)
- Sınıflandırma görevi için uyarlanmış YOLO mimarisi

**Zayıf Yönler:**
- Başlangıç eğitim kaybı çok yüksek (0.7518), yakınsama süresi daha uzun
- Eğitim kaybı diğer modellere kıyasla daha yüksek kalmıştır
- Doğrulama doğruluğu (%98.95) diğer modellerin gerisinde

---

## 4. Sınıf Bazlı Performans Karşılaştırması (Test Seti)

### 4.1 Hassasiyet (Precision) Karşılaştırması

| Sınıf | DenseNet-121 | ResNet-18 | YOLO26-N |
|-------|-------------|-----------|----------|
| Bazofil | **1.00** | 0.99 | **1.00** |
| Eozinofil | **1.00** | **1.00** | **1.00** |
| Eritroblast | **1.00** | **1.00** | **1.00** |
| Olgunlaşmamış Granülosit | 0.96 | 0.98 | **0.98** |
| Lenfosit | **1.00** | 0.99 | 0.99 |
| Monosit | 0.99 | 0.99 | 0.99 |
| Nötrofil | **1.00** | 0.98 | 0.99 |
| Trombosit | **1.00** | **1.00** | **1.00** |

### 4.2 Duyarlılık (Recall) Karşılaştırması

| Sınıf | DenseNet-121 | ResNet-18 | YOLO26-N |
|-------|-------------|-----------|----------|
| Bazofil | **1.00** | 0.99 | **1.00** |
| Eozinofil | **1.00** | **1.00** | **1.00** |
| Eritroblast | 0.99 | 0.99 | **1.00** |
| Olgunlaşmamış Granülosit | **0.99** | 0.97 | 0.98 |
| Lenfosit | **1.00** | **1.00** | 0.99 |
| Monosit | **1.00** | 0.98 | 0.99 |
| Nötrofil | 0.97 | 0.98 | 0.98 |
| Trombosit | **1.00** | **1.00** | **1.00** |

### 4.3 F1-Puanı (F1-Score) Karşılaştırması

| Sınıf | DenseNet-121 | ResNet-18 | YOLO26-N |
|-------|-------------|-----------|----------|
| Bazofil | **1.00** | 0.99 | **1.00** |
| Eozinofil | **1.00** | **1.00** | **1.00** |
| Eritroblast | 0.99 | **1.00** | **1.00** |
| Olgunlaşmamış Granülosit | **0.98** | 0.97 | **0.98** |
| Lenfosit | **1.00** | 0.99 | 0.99 |
| Monosit | 0.99 | 0.98 | 0.99 |
| Nötrofil | 0.98 | 0.98 | 0.98 |
| Trombosit | **1.00** | **1.00** | **1.00** |

---

## 5. Karışıklık Matrisi Analizi

### DenseNet-121
- Bazofil, Eozinofil ve Trombosit sınıfları **hatasız** sınıflandırılmıştır.
- En fazla hata **Nötrofil** sınıfında gözlemlenmiştir (20 örnek yanlış sınıflandırılmış).
- Olgunlaşmamış Granülosit sınıfında 3 örnek Nötrofil olarak yanlış sınıflandırılmıştır.

### ResNet-18
- Bazofil sınıfında 2 örnek hatalı sınıflandırılmıştır.
- **Olgunlaşmamış Granülosit** sınıfında en fazla hata gözlemlenmiştir (13 örnek Nötrofil olarak sınıflandırılmış).
- Nötrofil sınıfında 10 örnek Olgunlaşmamış Granülosit olarak yanlış sınıflandırılmıştır.
- Olgunlaşmamış Granülosit ve Nötrofil arasındaki karışıklık en belirgin sorundur.

### YOLO26-N
- Eozinofil ve Trombosit sınıfları **hatasız** sınıflandırılmıştır.
- Olgunlaşmamış Granülosit sınıfında 8 örnek Nötrofil olarak yanlış sınıflandırılmıştır.
- Nötrofil sınıfında 13 örnek Olgunlaşmamış Granülosit olarak yanlış sınıflandırılmıştır.
- Monosit sınıfında 2 örnek hatalı sınıflandırılmıştır.

**Ortak Bulgu:** Üç modelde de **Olgunlaşmamış Granülosit** ve **Nötrofil** sınıfları arasında en fazla karışıklık yaşanmaktadır. Bu, iki hücre tipinin morfolojik benzerliğinden kaynaklanmaktadır.

---

## 6. Eğitim Kararlılığı Analizi

| Metrik | DenseNet-121 | ResNet-18 | YOLO26-N |
|--------|-------------|-----------|----------|
| Doğrulama Kaybı Dalgalanması | Düşük | **Yüksek** | Orta |
| Yakınsama Hızı | Hızlı (Epoch 1'de %98.66) | Orta | Yavaş (Epoch 8'de %98+) |
| Overfitting Riski | Düşük | Orta | **En Düşük** |
| Eğitim Kararlılığı | **En İyi** | En Zayıf | İyi |

---

## 7. Genelleme Kapasitesi

Genelleme kapasitesi, doğrulama doğruluğu ile test doğruluğu arasındaki fark üzerinden değerlendirilmiştir:

| Model | Doğrulama Doğruluğu (En İyi) | Test Doğruluğu | Fark |
|-------|-----------------|----------|------|
| DenseNet-121 | 99.42% | 99.09% | -0.33% |
| ResNet-18 | 99.36% | 98.89% | -0.47% |
| YOLO26-N | 98.95% | 99.12% | **+0.17%** |

- **YOLO26-N** test setinde doğrulamadan daha iyi performans göstermiştir - bu güçlü genelleme kapasitesine işaret eder.
- **ResNet-18** en yüksek performans kaybını yaşamıştır (-0.47%).
- **DenseNet-121** makul bir genelleme kaybı göstermiştir (-0.33%).

---

## 8. Sonuç ve Değerlendirme

### Genel Sıralama

| Kriter | 1. | 2. | 3. |
|--------|-----|-----|-----|
| Doğrulama Doğruluğu | DenseNet-121 (%99.42) | ResNet-18 (%99.36) | YOLO26-N (%98.95) |
| Test Doğruluğu | YOLO26-N (%99.12) | DenseNet-121 (%99.09) | ResNet-18 (%98.89) |
| Eğitim Kararlılığı | DenseNet-121 | YOLO26-N | ResNet-18 |
| Genelleme | YOLO26-N | DenseNet-121 | ResNet-18 |
| Hız/Hafiflik | ResNet-18 | YOLO26-N | DenseNet-121 |

### Öneriler

1. **En yüksek doğruluk için:** DenseNet-121 tercih edilmelidir. Hem doğrulama hem test setinde tutarlı yüksek performans göstermiştir.

2. **En iyi genelleme için:** YOLO26-N tercih edilmelidir. Test setinde en yüksek doğruluğu yakalamış ve overfitting riski en düşük modeldir.

3. **Hafif ve hızlı çözüm için:** ResNet-18 tercih edilmelidir. Daha az parametre ile rekabetçi sonuçlar üretmiştir.

4. **Tüm modellerde ortak sorun:** Olgunlaşmamış Granülosit ve Nötrofil sınıfları arasındaki karışıklık azaltılmalıdır. Bu sınıflar için veri artırma (data augmentation) veya sınıf-spesifik öğrenme stratejileri uygulanabilir.

### Genel Yorum

Üç model de BloodMNIST veri setinde **%98.89 - %99.12** test doğruluğu aralığı ile oldukça başarılı sonuçlar elde etmiştir. Modeller arasındaki farklar küçük olup, uygulama senaryosuna göre (hız, doğruluk, genelleme) uygun model seçilmelidir. Transfer learning yaklaşımı tüm modellerde etkili olmuş ve sınırlı veri ile yüksek performans elde edilmesini sağlamıştır.

---

*Bu analiz, BloodMNIST veri seti üzerinde DenseNet-121, ResNet-18 ve YOLO26-N modellerinin eğitim sonuçlarına dayanmaktadır.*
