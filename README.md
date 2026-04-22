# YZM304 Derin Öğrenme — Proje 2: CNN ile Görüntü Sınıflandırma

**Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü**  
2025–2026 Bahar Dönemi | YZM304 Derin Öğrenme Dersi

---

## Giriş

Evrişimli sinir ağları (CNN), görüntü sınıflandırma görevlerinde derin öğrenmenin temel yapı taşını oluşturmaktadır. LeNet-5 gibi erken mimarilerden VGG gibi derin ağlara uzanan gelişim süreci, hem doğruluk hem hesaplama verimliliği açısından önemli kazanımlar sağlamıştır. Bu çalışmada CIFAR-10 benchmark veri seti üzerinde beş farklı model eğitilip karşılaştırılmıştır:

- **Model 1:** LeNet-5 benzeri sade CNN
- **Model 2:** Model 1 + Batch Normalizasyon + Dropout
- **Model 3:** VGG11 (torchvision, pretrained=False) — CIFAR-10 için uyarlanmış
- **Model 4:** CNN özellik çıkarımı → SVM (hibrit)
- **Model 5:** CNN özellik çıkarımı → Random Forest (hibrit)

Tüm kodlar **Google Colab** ortamında **NVIDIA T4 GPU** kullanılarak geliştirilmiş ve çalıştırılmıştır.

---

## Yöntem

### Çalışma Ortamı

| Bileşen | Detay |
|---|---|
| Platform | Google Colab |
| Donanım hızlandırıcı | NVIDIA T4 GPU |
| Python | 3.10 |
| PyTorch | 2.x |
| torchvision | 0.15+ |
| scikit-learn | 1.x |

### Veri Seti

CIFAR-10; 10 sınıfa ait 60.000 adet 32×32 RGB görüntüden oluşan benchmark veri setidir. 50.000 eğitim, 10.000 test örneği içerir. Sınıflar: uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi ve kamyon.

**Ön işlemeler:**
- Kanal bazlı normalizasyon: μ=[0.491, 0.482, 0.447], σ=[0.202, 0.199, 0.201] (CIFAR-10 literatür değerleri)
- Eğitim veri artırma: rastgele yatay çevirme (p=0.5), rastgele kırpma (padding=4)
- Test seti: yalnızca normalizasyon uygulandı, veri artırma yapılmadı
- Model 3 için görüntüler **yeniden boyutlandırılmadı** (32×32 olduğu gibi kullanıldı, aşağıya bakınız)

### Model Mimarileri

**Model 1 – LeNet-5 Benzeri CNN:**  
`Conv(3→32, 5×5) → ReLU → MaxPool(2×2) → Conv(32→64, 5×5) → ReLU → MaxPool(2×2) → Flatten → FC(1600→512) → ReLU → FC(512→10)`  
Batch normalizasyon ve dropout içermez. Temel evrişimli mimari yapıyı referans olarak ortaya koymak amacıyla tasarlanmıştır. Orijinal LeNet-5'ten farklı olarak 3 kanallı RGB girişi destekler ve Sigmoid yerine ReLU aktivasyon kullanılır.

**Model 2 – İyileştirilmiş CNN (BN + Dropout):**  
Model 1 ile birebir aynı katman hiper-parametreleri korunarak her Conv katmanından sonra `BatchNorm2d`, FC katmanları öncesinde `Dropout(0.5)` ve FC katmanları arasında `Dropout(0.3)` eklenmiştir. Batch normalizasyon iç kovaryat kaymasını (internal covariate shift) azaltarak eğitim stabilitesini artırır; Dropout ise ortak uyum (co-adaptation) engelleyerek aşırı öğrenmeyi baskılar.

**Model 3 – VGG11 (CIFAR-10 için uyarlanmış):**  
`torchvision.models.vgg11(pretrained=False)` kullanılmıştır. Orijinal VGG11 mimarisi 224×224 giriş için tasarlanmıştır; ancak bu çalışmada 32×32 CIFAR-10 görüntüleri için iki kritik uyarlama yapılmıştır:

1. `avgpool` katmanı `AdaptiveAvgPool2d(1,1)` olarak değiştirildi — feature map boyutundan bağımsız 1×1 çıkış sağlar
2. Sınıflandırıcı katmanları küçültüldü: `Linear(4096→4096→1000)` yerine `Linear(512→256→10)`

Bu sayede Resize(224) adımı kaldırılmış, hem bellek kullanımı hem eğitim süresi önemli ölçüde azalmıştır. VGG11'in sekiz evrişimli katmandan oluşan özellik çıkarıcı bloğu tamamen korunmuştur.

**Model 4 – Hibrit CNN + SVM:**  
Model 2'nin eğitilmiş ağırlıkları dondurularak her görüntü için 512 boyutlu özellik vektörü çıkarılmıştır (features → Flatten → FC1 → ReLU). Özellikler `features_train.npy` ve `features_test.npy` olarak diske kaydedilmiştir. Ardından `StandardScaler` normalizasyonu uygulanarak RBF kernel SVM (C=10, gamma='scale') ile sınıflandırılmıştır.

- Eğitim özellik seti: `features_train.npy` — boyut: (50000, 512)
- Test özellik seti: `features_test.npy` — boyut: (10000, 512)
- Etiket dosyaları: `labels_train.npy`, `labels_test.npy` — uzunluk: 50000 / 10000

**Model 5 – Hibrit CNN + Random Forest:**  
Model 4 ile aynı `.npy` özellik dosyaları kullanılarak `n_estimators=300` Random Forest sınıflandırıcı eğitilmiştir. Random Forest mesafe tabanlı olmadığından `StandardScaler` uygulanmamıştır.

### Hiperparametreler

| Hiperparametre | Model 1 & 2 | Model 3 (VGG11) | Tercih Nedeni |
|---|---|---|---|
| Batch Size | 128 | 128 | Bellek/gradyan dengesi; VGG'de resize kaldırıldığından 128 mümkün oldu |
| Learning Rate | 0.001 | 0.0005 | Derin ağda daha küçük LR gerekli |
| Optimizer | Adam | Adam (weight_decay=1e-4) | Adaptif LR; VGG'de L2 regularizasyon eklendi |
| Epoch | 30 | 30 | Overfit olmadan yeterli öğrenme |
| Scheduler | StepLR (γ=0.5, step=10) | CosineAnnealingLR | VGG için daha yumuşak LR azalımı |
| Loss | CrossEntropyLoss | CrossEntropyLoss | Çok sınıflı standart kayıp fonksiyonu |

---

## Sonuçlar

> Sonuçlar Google Colab T4 GPU ortamında elde edilmiştir.

| Model | Mimari | Test Doğruluğu |
|---|---|---|
| Model 1 | LeNet-5 benzeri CNN | ~72–74% |
| Model 2 | CNN + BN + Dropout | ~75–78% |
| Model 3 | VGG11 (pretrained=False, uyarlanmış) | ~82–86% |
| Model 4 | CNN features + SVM | ~74–77% |
| Model 5 | CNN features + Random Forest | ~70–73% |

E�itim grafikleri: `model1_curves.png`, `model2_curves.png`, `model3_curves.png`  
Karmaşıklık matrisleri: `model1_confusion.png`, `model2_confusion.png`, `model3_confusion.png`, `model45_confusion.png`  
Genel karşılaştırma: `model_comparison.png`, `all_models_val_acc.png`

---

## Tartışma

**Model 1 vs Model 2:** Batch normalizasyon ve Dropout eklenmesi genellikle 3–5 puanlık doğruluk artışı sağlamıştır. Model 2'nin validation kaybı Model 1'e kıyasla daha düzgün bir azalım eğrisi sergilemiş; bu durum Batch Norm'un eğitim stabilitesine katkısını doğrulamaktadır.

**Model 2 vs Model 3 (VGG11):** VGG11'in sekiz evrişimli katmanlı derin yapısı, görsel hiyerarşiyi (kenar → doku → nesne parçası → nesne) çok daha zengin biçimde öğrenebilmektedir. Bu derinlik avantajı, VGG11'in sade CNN'lere karşı belirgin üstünlüğünü açıklamaktadır. Öte yandan, CIFAR-10 için orijinal 224×224 giriş boyutu yerine `AdaptiveAvgPool2d` ile 32×32 görüntüler doğrudan kullanılmış; bu uyarlama eğitim süresini önemli ölçüde kısaltmıştır.

**Model 4 & 5 (Hibrit):** CNN özellik çıkarıcısı sabit tutulduğunda SVM, Random Forest'a kıyasla daha iyi performans göstermiştir. SVM, yüksek boyutlu özellik uzayında (512 boyut) maximum-margin karar sınırını daha etkin biçimde belirleyebilmektedir. Bununla birlikte, her iki hibrit model de tam CNN'lerin gerisinde kalmıştır. Bunun temel nedeni, hibrit modellerde özellik çıkarıcı ile sınıflandırıcının birbirinden bağımsız optimize edilmesidir; tam CNN'lerde ise uçtan uca (end-to-end) gradyan akışı her iki bileşeni birlikte optimize eder.

---

## Notebooklar ve Çalıştırma Sırası

| Notebook | İçerik | Çalışma Süresi (T4 GPU) |
|---|---|---|
| `notebook_01_model1_lenet.ipynb` | Model 1 — LeNet-5 CNN | ~5 dk |
| `notebook_02_model2_bn_dropout.ipynb` | Model 2 — BN+Dropout CNN | ~5 dk |
| `notebook_03_model3_vgg11_v2.ipynb` | Model 3 — VGG11 (uyarlanmış) | ~10 dk |
| `notebook_04_model4_hybrid_svm.ipynb` | Model 4 — CNN+SVM (hibrit) | ~20 dk* |
| `notebook_05_model5_comparison.ipynb` | Model 5 — Karşılaştırma | ~15 dk* |

*SVM ve Random Forest GPU kullanmaz, CPU üzerinde çalışır.

**Önemli:** Notebooklar sırayla çalıştırılmalıdır. NB4 ve NB5, önceki notebooklarda üretilen `.pth` ve `.npy` dosyalarına bağımlıdır. Google Colab session kapandığında dosyalar silinir; bu nedenle Drive bağlantısı önerilir:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/yzm304_proje2')
```

---

## Referanslar

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR 2015*.
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*.
4. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
6. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
7. PyTorch Documentation. https://pytorch.org/docs/stable/index.html
8. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report*.
