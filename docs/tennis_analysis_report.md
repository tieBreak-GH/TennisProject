# Tenis Analiz Sistemi (TennisProject) - Detaylı Teknik İnceleme Raporu

Bu rapor, `TennisProject` projesinin derin öğrenme, makine öğrenmesi ve bilgisayarlı görü (computer vision) mimarisini teknik düzeyde incelemek amacıyla hazırlanmıştır.

---

## 1. Video Giriş ve Sahne Analizi Modülü
* **Bileşenler**: [main.py](file:///Users/gokhan/development/TennisProject/main.py), [utils.py](file:///Users/gokhan/development/TennisProject/utils.py#L6)
* **Çalışma Prensibi**: 
  * Girdi videosunun kareleri NumPy dizisi olarak belleğe alınır.
  * [scene_detect](file:///Users/gokhan/development/TennisProject/utils.py#L6) fonksiyonu, `PySceneDetect` kütüphanesini kullanarak renk histogramlarındaki ani değişimleri analiz eder ve videoyu bağımsız sahnelere (farklı kamera açılarına) böler.
  * **Amacı**: Tenis yayınlarında kamera açısı değiştiğinde veya tekrar gösterimler (replay) girdiğinde saha koordinatları değişir. Analizin sahne bazlı sınırlandırılması, perspektif dışı karelerdeki hatalı algılamaları önler.

---

## 2. Top Takip Algoritması (Ball Tracking)
* **Bileşenler**: [ball_detector.py](file:///Users/gokhan/development/TennisProject/ball_detector.py#L8), [tracknet.py](file:///Users/gokhan/development/TennisProject/tracknet.py#L16)
* **Kullanılan Model**: [BallTrackerNet](file:///Users/gokhan/development/TennisProject/tracknet.py#L16) (U-Net benzeri, Convolution-Deconvolution yapısı).
* **Çalışma Prensibi**:
  * Topun yüksek hızından kaynaklanan hareket ipuçlarını (motion cues) yakalayabilmek için ardışık 3 kare (frame) birleştirilerek modele beslenir. Bu sebeple giriş kanalı sayısı **9**'dur (`3 kare x 3 RGB kanalı`).
  * Görüntü `640x360` boyutuna küçültülür. Model çıktısı, topun olası konumunu gösteren bir ısı haritasıdır (heatmap).
  * **Post-processing**: Model çıktısı ikili eşikleme (`cv2.threshold`) ile filtrelenir ve `cv2.HoughCircles` kullanılarak top olabilecek dairesel nesneler aranır.
  * Önceki karedeki top konumu ile yeni adaylar arasındaki Öklid mesafesi hesaplanır. Mesafe eşiği (`max_dist=80`) aşıldığında ani sıçramalar elenerek top takibi kararlı hale getirilir.

---

## 3. Saha Tespiti ve Geometrik Hizalama (Homography)
* **Bileşenler**: [court_detection_net.py](file:///Users/gokhan/development/TennisProject/court_detection_net.py#L10), [homography.py](file:///Users/gokhan/development/TennisProject/homography.py#L17), [court_reference.py](file:///Users/gokhan/development/TennisProject/court_reference.py#L6), [postprocess.py](file:///Users/gokhan/development/TennisProject/postprocess.py#L22)
* **Kullanılan Model**: `BallTrackerNet` mimarisinin 15 çıkış kanallı versiyonu ([CourtDetectorNet](file:///Users/gokhan/development/TennisProject/court_detection_net.py#L10)).
* **Çalışma Prensibi**:
  * Ağ, saha üzerindeki 14 kritik köşe ve kesişim noktasını (keypoint) tespit eder (15. kanal yardımcı orta noktadır).
  * **Hassasiyet İyileştirme**: [refine_kps](file:///Users/gokhan/development/TennisProject/postprocess.py#L22) fonksiyonu, modelin bulduğu noktaların etrafında `40x40` piksellik bir kesim yapar, çizgileri `HoughLinesP` ile tespit eder ve çizgilerin kesişim noktasını matematiksel olarak (`SymPy.Line.intersection`) hesaplar. Bu işlem **piksel altı (sub-pixel) hassasiyetle** köşe tespiti sağlar.
  * **Homografi Matrisi**: [get_trans_matrix](file:///Users/gokhan/development/TennisProject/homography.py#L17) fonksiyonu, 14 noktadan elde edilen kombinasyonları test ederek [CourtReference](file:///Users/gokhan/development/TennisProject/court_reference.py#L6) şablonuna eşleyen en uygun perspektif dönüşüm matrisini (Homography) bulur. Bu matris yardımıyla kamera görüntüsündeki 2D piksel koordinatları, gerçek 3D dünya düzlemine (minimap koordinatlarına) yansıtılır.

---

## 4. Oyuncu Algılama ve Takibi (Player Detection)
* **Bileşenler**: [person_detector.py](file:///Users/gokhan/development/TennisProject/person_detector.py#L10)
* **Kullanılan Model**: PyTorch torchvision kütüphanesinden hazır eğitilmiş `Faster R-CNN ResNet-50 FPN`.
* **Çalışma Prensibi**:
  * Sadece "Person" (sınıf ID: 1) etiketli nesneler algılanır ve oyuncunun ayaklarının bastığı nokta (sınırlayıcı kutunun alt-orta noktası: `[int((x1+x2)/2), y2]`) hesaplanır.
  * Saha maskeleri (`ref_top_court` ve `ref_bottom_court`) homografi matrisi kullanılarak kamera perspektifine göre bükülür (`warpPerspective`).
  * Oyuncunun ayak koordinatının bu bükülmüş maskelerin içine düşüp düşmediği kontrol edilerek oyuncular "üst yarı saha" ve "alt yarı saha" olarak sınıflandırılır.
  * `filter_players` aktifse, sahada oyuncular dışındaki top toplayıcılar veya hakemler elenerek sadece sahaya en yakın olan 2 ana oyuncu takipte tutulur.

---

## 5. Sekme Tespiti (Bounce Detection)
* **Bileşenler**: [bounce_detector.py](file:///Users/gokhan/development/TennisProject/bounce_detector.py#L7)
* **Kullanılan Model**: **CatBoostRegressor**.
* **Çalışma Prensibi**:
  * Topun hareket yörüngesinde (trajectory) oluşan hız ve yön değişimlerini (fiziksel ivmelenmeleri) analiz eder.
  * **Veri Düzgünleştirme**: Topun görünmediği veya modelin kaçırdığı kareler `CubicSpline` interpolasyonuyla doldurulur ve top yörüngesi pürüzsüzleştirilir.
  * **Özellik Mühendisliği (Feature Engineering)**: Her kare için topun X ve Y koordinatlarının zaman serisi farkları (`diff`), geriye dönük gecikmeleri (`lag`), ileriye dönük farkları (`diff_inv`) ve bunların birbirine oranları (`y_div`, `x_div`) hesaplanır.
  * CatBoost modeli bu fizik tabanlı ivme ve yön özelliklerini alarak topun yere çarpma olasılığını tahmin eder.

---

## Mimari Eleştiri ve İyileştirme Fırsatları
1. **İşlem Hızı Darboğazı**: Her karede Faster R-CNN çalıştırılması CPU üzerinde aşırı yavaştır. Her $N$ karede bir tespit yapıp ara karelerde basit nesne takibi (tracking) kullanılması hızı büyük oranda artıracaktır.
2. **Saha Çizgi Titremeleri**: Saha çizgileri her karede bağımsız hesaplandığı için titreşime sebep olmaktadır. Sahne bazlı homografi matrisi ortalaması veya Kalman filtresi ile bu çizgiler tamamen kararlı hale getirilebilir.
