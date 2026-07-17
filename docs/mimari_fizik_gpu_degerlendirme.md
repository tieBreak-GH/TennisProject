# TennisProject — Mimari, Fizik/Mantık ve GPU Değerlendirme Raporu

*Hazırlanma tarihi: 16 Temmuz 2026. Kapsam: (1) mimari + teknik değerlendirme, (2) top hızı ölçümünün fizik/mantık analizi ve farklı kamera yüksekliği sorunu, (3) iyileştirme önerileri, (4) RX 9070 XT (AMD) ekran kartının video işlemede kullanımı.*

---

## 0. Yönetici Özeti

- **Mimari sağlam.** İki geçişli streaming pipeline, sahne-başı homografi önbellekleme, threaded okuma/yazma ve merkezî `config.py` iyi mühendislik. Bu katman üzerine söylenecek fazla bir şey yok; önceki raporlar (`senior_engineer_review.md`, `development_plan.md`) buradaki teknik borçların çoğunu zaten kapatmış.
- **En kritik bulgu fizikte:** Top hızı, topun **görüntüdeki pikselinin saha zemini homografisine** yansıtılmasıyla hesaplanıyor. Homografi yalnızca **zemin düzlemindeki (z=0)** noktalar için geçerlidir; top ise havada (z = h > 0). Bu yüzden ölçülen hız, kamera yüksekliğine bağlı sistematik bir hata içerir. **Kamera alçaldıkça hata patlar** (2–3 m'lik tripodda %100–300 abartıya, hatta sonsuza/çöp değere kadar). Bu, kullanıcının "farklı kamera yüksekliğinde yanlış hız" gözleminin tam matematiksel açıklamasıdır.
- **Doğru çözüm mümkün ve mevcut altyapının üstüne oturuyor:** Saha homografisi zaten kameranın 3B pozunu içeriyor. Bundan kamera yüksekliği türetilip, sekmeler-arası her uçuşa **projectile (parabol) fiziği** uydurularak topun gerçek 3B yörüngesi geri kazanılabilir. Bu, hızı **kamera yüksekliğinden bağımsız** hale getirir (Bölüm 4-C).
- **GPU sorununun kök nedeni bulundu:** Kurulu PyTorch **CPU-only derleme** (`torch 2.8.0+cpu`; CUDA yok, ROCm/HIP yok) ve `torch_directml` kurulu değil. Yani şu an **hiçbir** ekran kartı kullanılamıyor. RX 9070 XT (AMD RDNA4) için Windows'ta üç yol var (Bölüm 5); en kolayı `torch-directml`, en performanslısı WSL2 + ROCm.

---

## 1. Mimari Değerlendirme

### 1.1 Genel yapı
Pipeline temiz katmanlara ayrılmış:

| Katman | Dosya | Sorumluluk |
|---|---|---|
| Top tespiti | `ball_detector.py` + `tracknet.py` | TrackNet U-Net, 3-kare kayan pencere, heatmap→Hough |
| Saha tespiti | `court_detection_net.py` + `homography.py` + `postprocess.py` | 14 keypoint CNN, homografi çözümü, çizgi rafinasyonu |
| Sekme tespiti | `bounce_detector.py` | CatBoost regresör, yörünge lag özellikleri |
| Oyuncu tespiti | `person_detector.py` | YOLO11n + saha maskesiyle üst/alt filtreleme |
| Hız | `speed_estimator.py` | Homografi tabanlı km/h |
| Analiz | `rally_analyzer.py` | Ralli/vuruş segmentasyonu, servis + çizgi kararı |
| Orkestrasyon | `main.py` | İki geçişli streaming, cihaz seçimi, render |
| Arayüz | `app.py` | Streamlit batch UI |

**Güçlü yönler:** Sorumluluklar net; `config.py` tüm eşikleri tek yerde toplamış; `main.py`'deki iki geçişli streaming belleği video uzunluğundan bağımsız sabit tutuyor; `ThreadedFrameReader`/`ThreadedFrameWriter` decode/encode'u inference ile örtüştürüyor. Bu mimari, bu ölçekteki bir proje için beklenenin üzerinde.

### 1.2 Mimari düzeyinde iyileştirilebilecekler
- **Ölçüm/kalibrasyon katmanı yok.** Hız, homografi çıktısının üzerine doğrudan kuruluyor; araya bir "3B yörünge / kalibrasyon" katmanı girmiyor. Fizik hatasının (Bölüm 3) mimari kök nedeni budur — hız hesabı `speed_estimator.py`'de 2B zemin projeksiyonuna sıkışmış. Bölüm 4-C'deki çözüm, bu katmanı ekler ve `speed_estimator`'ı onun tüketicisi yapar.
- **Cihaz seçimi kod içine gömülü.** `_select_device` mantıklı ama arayüzden manuel cihaz seçme (ör. "CPU'ya zorla" / "GPU dene") yok. GPU deneysel backend'lerde (DirectML) sorun çıkarsa kullanıcının geri düşecek düğmesi yok.
- **`filter_players` ölü kodu** hâlâ duruyor (senior raporunda da not edilmiş) — kaldırılmalı ya da açıkça "kullanılmıyor" işaretlenmeli.

---

## 2. Teknik Değerlendirme

### 2.1 Doğru yapılmış teknik kararlar
- **Sahne-başı homografi önbellekleme** (`court_detection_net.infer_model` / `analyze_streaming`): kamera sahne içinde sabit varsayılıp homografi bir kez çözülüyor. Büyük kazanç, doğru varsayım.
- **Batch inference yalnızca CUDA'da** (`_infer_batch_size`): CPU/MPS'te batch'in yavaşlattığı ölçülmüş ve devre dışı bırakılmış — ampirik, doğru.
- **`avc1`/H.264 writer**: tarayıcı önizlemesi için gerekli; DIVX'ten geçiş doğru.
- **`fps` float olarak korunuyor**: 23.976 → 23'e budanma hatası düzeltilmiş.

### 2.2 Teknik riskler / dikkat noktaları
- **`cv2.VideoWriter(avc1)` sessiz başarısızlık riski:** Bazı OpenCV/FFmpeg derlemelerinde `avc1` encoder yoksa `VideoWriter` açılamaz ama exception fırlatmaz — boş/bozuk video üretir. `_open_writer`'da `writer.isOpened()` kontrolü ve anlaşılır hata eklenmeli. (Bu makinede FFmpeg mevcut ve çalışıyor; risk taşınabilirlikte.)
- **`BALL_SPEED_MAX_KMH = 300` sınırı fizik hatasıyla etkileşiyor:** Gerçek 150 km/h'lik bir vuruş, alçak kamera abartısıyla (Bölüm 3) 300'ü aşıp **sessizce `None`'a** düşebilir → hız grafiğinde boşluk. Yani sınır, hatayı gizlemek yerine veriyi siliyor.
- **`opencv-python==5.0.0.93` pini:** OpenCV 5 yeni bir major sürüm; bu makinede kuruldu ve çalışıyor, ancak farklı platform/Python'larda wheel bulunamama riski taşır. En azından README'de not düşülmeli.
- **Çift/üçlü decode:** Sahne kesme artık Pass 1'e gömülü (iyi); ama highlights/rallies-only render Pass 2 içinde. Bu bilinçli ve kabul edilebilir.

---

## 3. Fizik & Mantık: Top Hızı Neden Kamera Yüksekliğine Bağlı? (EN KRİTİK BÖLÜM)

### 3.1 Kök neden: düzlemsel homografi, havadaki topu zeminde sanır

`speed_estimator.get_ball_speed` her karede şunu yapıyor:

```python
pt_i = _transform_point(ball_track[i], homography_matrices[i])   # piksel → saha cm
dist_cm = distance.euclidean(pt_i, pt_j)
speed = (dist_cm/100) / dt * 3.6
```

`homography_matrices[i]`, görüntü pikselini **saha zemini düzlemine (z=0)** eşleyen bir homografidir. Bu eşleme **yalnızca fiziksel olarak zeminde olan noktalar için doğrudur.** Oyuncunun ayak noktası zemindedir (`person_detector` bilinçli olarak bbox alt-orta noktasını, yani ayakları kullanır — bu yüzden **oyuncu minimap konumu metrik olarak doğrudur**). Ama top havadadır: uçuş boyunca z = h > 0. Topun piksel konumunu zemin homografisinden geçirmek, topun gerçek zemin-izdüşümünü değil, **kameradan topa giden ışının zeminle kesiştiği noktayı** verir.

### 3.2 Hatanın türetimi

Kamera merkezi C = (cₓ, c_y, H) (H = kameranın saha üzerindeki yüksekliği). Top B = (bₓ, b_y, h). Kameradan B'ye giden ışının zemine (z=0) çarptığı nokta:

```
P(t) = C + t·(B − C),   z: H + t·(h − H) = 0  ⟹  t* = H / (H − h)
G = C + t*·(B − C)
```

Topun **gerçek** zemin izdüşümü F = (bₓ, b_y). Hata vektörü:

```
E = G − F = (B_xy − C_xy) · h / (H − h)
```

Yani homografi, topu kameranın zemin izdüşümünden **radyal olarak dışarı**, kamera-top yatay mesafesinin **h/(H−h)** katı kadar kaydırır.

**Hız için asıl önemli olan büyütme faktörü.** Top sabit h yüksekliğinde yatay hareket ederse, ölçülen zemin hızı:

```
v_ölçülen = M · v_gerçek,   M = H / (H − h)
```

M > 1 olduğundan **hız daima abartılır** ve abartı miktarı **kamera yüksekliği H'ye bağlıdır.** İşte kullanıcının gördüğü şey budur.

### 3.3 Sayısal etki (top yüksekliği h ≈ 1.5 m alındığında)

| Kamera yüksekliği H | Büyütme M = H/(H−h) | Hız hatası |
|---:|---:|---:|
| 20 m (yayın kulesi) | 1.08 | **+8%** |
| 12 m (yüksek yayın) | 1.14 | **+14%** |
| 8 m (tribün üstü) | 1.23 | **+23%** |
| 5 m (yüksek tripod/direk) | 1.43 | **+43%** |
| 3 m | 2.00 | **+100%** |
| 2.5 m (yüksek tripod) | 2.50 | **+150%** |
| 2.0 m (normal tripod) | 4.00 | **+300%** |
| ≤ 1.5 m (topla aynı hiza) | → ∞ | **çöp / tekillik** |

Ek olarak: top kameradan **yüksekteyse** (h > H — alçak telefonun üstünden geçen servis/lob), t* negatife döner, izdüşüm kameranın arkasına düşer → tamamen anlamsız, dev outlier. Bunlar `max_speed_kmh=300` sınırınca kısmen atılır ama medyan yumuşatmayı da bozar.

**Sonuç:** Yayın videolarında (yüksek kamera) hata küçük (%8–15) olduğu için bu projenin dayandığı önceden-eğitilmiş modeller ve tipik demolar "makul" görünür. Ama telefonu tripoda koyup çeken kullanıcıda (2–3 m) hata %100–300'e çıkar ve **kararsızdır** — aynı vuruş farklı yükseklikte farklı ölçülür. Bu bir bug değil, **yöntemin temel sınırı**.

### 3.4 İkincil (kamera yüksekliğinden bağımsız) hatalar
- **Dikey hız bileşeni yok sayılıyor.** 2B zemin projeksiyonu yalnızca yatay hızı yakalar; topun düşey hızı düşer. Düz sert servislerde küçük, lob/yüksek toplarda büyük → gerçek 3B hızın altında ölçüm. Bu hata yükseklikten bağımsızdır ama 3.2'deki büyütmeyle **ters yönde** çalışıp durumu daha da karıştırır.
- **Kiriş (chord) vs yay:** 5 karelik pencerede (~0.2 s @24fps) topun eğri yolu düz çizgiyle ölçülür → hafif düşük ölçüm. Sekme sınırında pencerenin kesilmesi bunu kısmen giderir.

---

## 4. Farklı Kamera Yüksekliğinde Doğru Hız — Çözüm Seçenekleri

Seçenekler ucuz→doğru sırasında. **C tavsiye edilir.**

### 4-A. Sadece sekme anında ölç (ucuz, kısmi)
Sekme karesinde h = 0'dır, homografi **tam olarak geçerlidir.** Sekmenin bir kare öncesi/sonrasıyla ölçülen hız o an için metrik olarak doğrudur. Ama sekme hızı ≠ tepe/servis hızı (top yavaşlamış ve yön değiştirmiştir). Değeri sınırlı ama dürüst; hızlı bir "güven göstergesi" olarak eklenebilir.

### 4-B. Homografiden kamera yüksekliğini çıkar, varsayılan top yüksekliğiyle düzelt (orta)
Saha homografisi H_mat = K·[r₁ r₂ t] ayrıştırılarak kamera pozu (dolayısıyla **yüksekliği**) metrik olarak elde edilebilir — çünkü `CourtReference` gerçek cm cinsindendir (baseline–baseline 2374 birim ≈ 2377 cm doğrulandı). K (odak uzaklığı) saha çizgilerinin kaçış noktalarından tahmin edilebilir (kullanıcı girdisi gerekmez). Sonra kare başına M = H/(H−h) faktörü, varsayılan bir top-yükseklik profiliyle bölünür. B'den daha iyidir ama hâlâ bir yükseklik modeli varsayar.

### 4-C. Tek kameradan 3B yörünge + projectile fiziği (DOĞRU ÇÖZÜM) ✅ tavsiye
Kamera yüksekliğinden **bağımsız**, spor-görü literatüründe yerleşik yöntem. Mevcut altyapının (homografi + sekme segmentasyonu) üstüne oturur:

1. **Kamerayı sahadan kalibre et.** Homografi → kamera pozu; odak uzaklığı saha çizgilerinin kaçış noktalarından. Artık her piksel, bilinen kamera merkezinden çıkan metrik bir 3B **ışın** tanımlar.
2. **Her uçuş segmentinde (vuruş→sekme, sekme→sekme) top parabol izler:**
   `p(t) = p₀ + v₀·t + ½·g·t²`,  g = (0, 0, −9.81 m/s²). 6 bilinmeyen (p₀, v₀).
3. **Her 2B top tespiti, 3B noktayı kendi ışınına kısıtlar.** p₀, v₀ en küçük karelerle (`scipy.optimize.least_squares`) çözülür; segmentte ~5+ kare varsa sistem aşırı-belirli ve çözülebilir.
4. Çıktı: her an için **gerçek 3B konum ve hız** → kamera yüksekliğinden bağımsız gerçek hız, ayrıca gerçek sekme konumları, tepe yüksekliği, vuruş/çarpma hızları.

Bu, projeyi "göreli hız göstergesi"nden "metrik hız ölçeri"ne yükseltir. Eklenecekler: kaçış-noktasından odak tahmini, homografi ayrıştırma, segment başına parabol fit. Orta çaba, yüksek getiri — ve **kullanıcının sorusunun doğrudan cevabı budur.**

> **Uygulama sırasında doğrulanan sınırlama** (bkz. `docs/uygulama_plani.md` Faz 3.4, kod: `trajectory_3d.py`): tek kameradan 3B çözümün doğruluğu, topun hareket yönünün kameranın bakış eksenine göre açısına ve segment uzunluğuna bağlıdır. Top kameradan **doğrudan uzaklaşıyor/yaklaşıyorsa** (radyal hareket — tenis'te sahayı boyluca kat eden vuruşlarda sık), piksel gürültüsü ışın-boyunca derinliğe/hıza yansır ve reprojeksiyon hatası (`rmse_px`) bunu **gizler** (ölçülen: %64 hızlı hata, `rmse_px` yine de düşük). Kısa uçuş segmentleri de benzer şekilde güvenilmez. Bu, yöntemi geçersiz kılmaz — kovaryans tabanlı bir belirsizlik metriğiyle (`speed_std_kmh`) güvenilir/güvenilmez segmentler ayırt edilip güvenilmeyenlerde 2B'ye düşülüyor — ama "her segmentte 3B çalışacak" beklentisini "yeterince uzun ve/veya yeterince enine segmentlerde 3B, diğerlerinde 2B fallback" olarak düzeltir.
>
> **Gerçek maç videosuyla doğrulanan, daha ciddi bir sınırlama** (Faz 5.1, bkz. `docs/uygulama_plani.md` §5.1): kullanıcının sağladığı gerçek bir yayın klibinde (uzun odaklı, sahanın arkasından, yükseltilmiş klasik TV kamerası — en yaygın tenis yayın açısı) **19 sekme-arası segmentin tamamı** fiziksel olarak imkânsız bir 3B çözüme yakınsadı (300 km/h – 3×10⁸ km/h arası), ve bunlardan **3 tanesi** kovaryans-tabanlı `speed_std_kmh` eşiğini de geçerek "güvenilir" göründü — çünkü bu metrik yalnızca yerel (LM çözücüsünün yakınsadığı noktadaki) eğriliği yakalıyor, küresel kötü-koşullanmayı değil. Bu videonun kamera açısı neredeyse tüm ralinin radyal hareket olması anlamına geliyor: yani bu **en yaygın** yayın kurulumu, yöntemin en zayıf olduğu durumla örtüşüyor. Düzeltme olarak `is_reliable_fit`'e sabit bir fiziksel hız tavanı (300 km/h) eklendi; bu videoda sonuç **3B'nin 0 karede devreye girmesi** (tamamen 2B'ye düşüş) oldu — yani yöntem kendini imkânsız çıktı üretmekten korudu ama bu kamera açısı için orijinal yükseklik-bağımlılığı sorununu **çözemedi**. 3B yöntemin gerçek faydası şu an için daha fazla yanal paralaksı olan kamera açılarıyla (yandan/köşeden, daha alçak/geniş açı) sınırlı; klasik uzun-odaklı arka-taraf yayın kamerası için ek iş (çoklu-segment fiziksel önseller, ya da yeterli paralaksı olmayan sahnede 3B'yi hiç denememe) gerekiyor.

### 4-D. Stereo / çok kamera (Hawk-Eye tarzı) — kapsam dışı, referans
İki senkron kamera 3B'yi doğrudan üçgenler. En doğru yöntem (altın standart) ama farklı çekim kurulumu gerektirir. Bağlam için anıldı.

### 4-E. Bugün için pratik çekim önerisi (kod değişmeden hatayı azaltır)
`M = H/(H−h)` faktörünü 1'e yaklaştırmak için: **kamerayı olabildiğince YÜKSEK ve GERİDE** konumlandırın (baseline arkası, yükseltilmiş). Bu hem M'yi küçültür hem topu daima kameranın altında tutar (tekillikten kaçınır). README'deki "Video çekim önerileri" bölümüne **yükseklik** maddesi eklenmeli (şu an sadece "sabit olsun" diyor).

---

## 5. RX 9070 XT (AMD) Ekran Kartı — Video İşlemede Kullanım

### 5.1 Kök neden (doğrulandı)
Bu ortamda kurulu olan PyTorch **CPU-only derleme**:
```
torch 2.8.0+cpu   ·   cuda_available = False   ·   cuda_built = None   ·   hip = None
torch_directml → kurulu değil
```
`requirements.txt`'teki düz `torch==2.8.0` pini, Windows'ta PyPI'dan **varsayılan olarak CPU wheel'ini** çeker. Yani `main._select_device` her koşulda `'cpu'`'ya düşüyor — kartınız (ya da herhangi bir kart) hiç denenmiyor bile. İlk yapılacak: **GPU-destekli bir torch derlemesi kurmak.**

### 5.2 Önemli ayrım: "video işleme" ≠ tek parça
- **Sinir ağı çıkarımı** (top/saha/YOLO): GPU'dan faydalanır — asıl darboğaz budur (geliştirme planına göre decode toplam sürenin %0.4'ü). **Kazanç buradadır.**
- **Video decode/encode** (`cv2.VideoCapture`/`VideoWriter`): FFmpeg ile **CPU'da** çalışır; DirectML/ROCm bunu hızlandırmaz. RX 9070 XT'nin donanımsal video motoru (VCN) ayrıca `ffmpeg h264_amf/av1_amf` ile encode'u hızlandırabilir ama OpenCV'nin paket FFmpeg'i bunu kullanmaz (ayrı iş). Neyse ki bu kısım zaten darboğaz değil.

### 5.3 RX 9070 XT için Windows'ta yollar (kolaydan performanslıya)

| Yol | Nasıl | Artı / Eksi |
|---|---|---|
| **1. torch-directml** (en kolay) | Ayrı bir venv'de `pip install torch-directml`. Kod `_select_device` içinde zaten otomatik algılıyor. | + Kurulum kolay, DX12'li her AMD kartı görür. − Daha eski bir torch'a düşürür (Python ≤3.12 ile uyumlu), performans orta, bazı op'lar desteklenmeyebilir. RDNA4 yeni olduğundan test şart. |
| **2. WSL2 + ROCm** (en iyi denge) | WSL2 Ubuntu kur → ROCm PyTorch wheel'i (`--index-url .../rocm6.x`). README zaten "AMD on Linux (ROCm) sorunsuz" diyor. | + Native'e yakın performans. − RDNA4 (gfx1201) desteği ROCm 6.4+ gerektirir; kurulum WSL2 GPU passthrough ister. |
| **3. Native ROCm-on-Windows** | AMD'nin Windows ROCm PyTorch önizleme wheel'leri (2025'te çıkmaya başladı). | + Yerel, WSL yok. − Deneysel/bleeding-edge, RDNA4 desteği sürüme bağlı. |
| **4. Linux dual-boot** | Ubuntu + ROCm PyTorch. | + En temiz, en hızlı ROCm yolu. − Ayrı işletim sistemi. |

**Öneri:** Önce **Yol 1 (torch-directml)** ile hızlı bir doğrulama yapın (kart görünüyor mu, çıktı doğru mu). Performans yetersizse **Yol 2 (WSL2 + ROCm)**'ye geçin. Her iki durumda da mevcut kod değişikliği gerekmez — `_select_device` backend'i otomatik seçiyor; tek gereken doğru torch derlemesinin kurulu olması.

**Not:** `requirements.txt`'e GPU torch'u koymayın (platforma göre değişir, CI'yı kırar). Bunun yerine README'ye kısa bir "AMD Windows GPU kurulumu" bölümü + arayüze manuel cihaz seçici (Bölüm 1.2) eklemek doğru yaklaşımdır.

---

## 6. Önceliklendirilmiş Geliştirme Önerileri

| # | Öneri | Kategori | Etki | Çaba |
|:--:|---|:--:|:--:|:--:|
| 1 | **3B projectile yörünge ile hız** (Bölüm 4-C) — hızı kamera yüksekliğinden bağımsız kıl | Fizik | **Çok yüksek** | Orta |
| 2 | **GPU'yu etkinleştir**: torch-directml/ROCm kurulum rehberi + arayüzde cihaz seçici (Bölüm 5) | Teknik | Yüksek | Kolay |
| 3 | **Çekim rehberine "yükseklik" maddesi** + arayüzde hızın "tahmini" olduğu ve kamera yüksekliğine duyarlılığı uyarısı (Bölüm 4-E) | UX/Doğruluk | Yüksek | Çok kolay |
| 4 | **`VideoWriter.isOpened()` kontrolü** `_open_writer`'da (Bölüm 2.2) | Teknik | Orta | Çok kolay |
| 5 | **`max_speed_kmh` sınırının veriyi silmesini düzelt** (Bölüm 2.2) — outlier'ı ayrı işaretle | Mantık | Orta | Kolay |
| 6 | **Ara adım olarak 4-A (sekmede metrik hız)** — 4-C'den önce hızlı, dürüst bir referans değer | Fizik | Orta | Kolay |
| 7 | **`filter_players` ölü kodunu temizle** | Bakım | Düşük | Çok kolay |

**Kısa vadede en yüksek getiri:** 2 + 3 + 4 (hepsi kolay, kullanıcı deneyimini ve doğruluğu hemen etkiler). **Projeyi asıl dönüştürecek olan:** 1 numaralı madde (3B yörünge) — kullanıcının "farklı kamera yüksekliğinde doğru hız" sorusunun kalıcı çözümü.
