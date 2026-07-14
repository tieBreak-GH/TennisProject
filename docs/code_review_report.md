# TennisProject — Senior Kod İncelemesi Raporu

**Tarih:** 14 Temmuz 2026 (son güncelleme: aynı gün, düzeltmeler uygulandıktan sonra)
**Kapsam:** Proje kök dizinindeki tüm Python kaynak dosyaları (venv/weights hariç)
**Genel değerlendirme:** Araştırma/prototip kalitesinde, çalışan bir bilgisayarlı görü pipeline'ı. Mimari kurgu (sahne tespiti → top takibi → saha tespiti → oyuncu tespiti → sekme tespiti → görselleştirme) doğru düşünülmüş.

**Durum:** Bu raporda listelenen 5 kritik bulgunun tamamı ve yüksek öncelikli bulguların çoğu düzeltildi ve gerçek video/model çalıştırmalarıyla doğrulandı (bkz. her madde altındaki "✅ Düzeltildi" notları). Ayrıca yeni bir özellik olarak top hızını topun yanına yazan `speed_estimator.py` eklendi (homografi üzerinden gerçek saha düzlemine projeksiyonla km/h hesabı). Kalan açık maddeler §2.1, §2.3, §3, §4'te işaretlenmiştir.

---

## 1. Kritik Bulgular (çökme veya sessiz yanlış sonuç riski) — hepsi düzeltildi

### 1.1 Sekme (bounce) karesinde top koordinatı `None` ise program çöker
`main.py:95-98` — `if i in bounces and inv_mat is not None` koşulunda `ball_track[i]`'nin geçerli olduğu kontrol edilmiyordu. Sekme tespit edilen karede `ball_track[i] = (None, None)` olabilir ve `np.array((None, None), dtype=np.float32)` `TypeError` fırlatırdı.

**✅ Düzeltildi:** `main.py`'de bounce çizim koşuluna `ball_track[i][0] is not None` guard'ı eklendi.

### 1.2 Sabit `scale=2` — 1280x720 dışındaki videolarda sessizce yanlış koordinat
`ball_detector.py:44` ve `court_detection_net.py:22` — model çıktısı 640x360'tan orijinal çözünürlüğe sabit 2 katsayısıyla ölçekleniyordu; 1920x1080 gibi bir girişte top ve saha noktaları tamamen yanlış yere çiziliyordu.

**✅ Düzeltildi:** Her iki dosyada da `scale_x = orig_width/640`, `scale_y = orig_height/360` dinamik hesaba geçildi (kare boyutlarını olmayan aspect-ratio'larda da doğru ele alacak şekilde x/y ayrı ayrı). 1920x1080 sentetik kareyle uçtan uca test edildi, koordinatlar beklenen aralıkta üretildi.

### 1.3 Truthiness ile `None` kontrolü: `x == 0` koordinatı kaybolur
`if ball_track[i][0]:` (`main.py`), `if prev_pred[0]:` (`ball_detector.py`), `if not x_ball[num]:` (`bounce_detector.py`) — koordinat `0`/`0.0` olduğunda bu kontroller `None` ile aynı davranıyordu.

**✅ Düzeltildi:** Üç dosyada da `is None`/`is not None` kontrollerine geçildi. Somut test: top `x=0.0` konumundayken eski kod bunu "eksik tespit" sanıp ekstrapolasyonla eziyordu; düzeltmeden sonra ezmiyor (doğrulandı).

### 1.4 `np.mean([])` → `nan` (homography.py:35)
Bir konfigürasyonun 4 noktası bulunmuş ama geriye kalan hiçbir nokta tespit edilememişse `dists` boş kalıyor, `np.mean([])` `RuntimeWarning` ile `nan` dönüyordu.

**✅ Düzeltildi:** `if not dists: continue` guard'ı eklendi; değişkenler `mean_error`/`best_error` olarak yeniden adlandırıldı. **Düzeltme sırasında test edilen gerçek etki:** `nan < inf` zaten Python/numpy'de her zaman `False` döndüğü için orijinal kod yanlış matris seçimine yol açmıyordu — sadece gürültülü `RuntimeWarning` üretiyordu. Rapordaki ilk ifade bu açıdan fazla iddialıydı; düzeltme yine de doğru bir temizlik (log gürültüsünü önlüyor).

### 1.5 `PersonDetector(device)` — `dtype` parametresine device string geçiliyor
`person_detector.py:11` imzası `def __init__(self, dtype=torch.FloatTensor)` idi, ama `main.py`'da `PersonDetector(device)` çağrılıyordu; `'cuda'` stringi `.to()` tarafından tesadüfen kabul ediliyordu.

**✅ Düzeltildi:** Parametre `device='cpu'` olarak yeniden adlandırıldı, `self.dtype` → `self.device`, tüm `.to()` çağrıları güncellendi.

---

## 2. Yüksek Öncelikli Bulgular

### 2.1 Tüm video RAM'e yükleniyor — ⏳ açık
`main.py:12-23` — `read_video` her kareyi listede tutuyor. 720p, 30fps, 5 dakikalık video ≈ 9000 kare × ~2.7 MB ≈ **24 GB** ham bellek. Kısa kliplerde çalışır, gerçek maç videosunda OOM kaçınılmaz.

**Öneri (değişmedi):** Pipeline'ı akış (streaming) mimarisine geçirin: kareleri sahne sahne oku-işle-yaz.

### 2.2 `torch.no_grad()` kullanılmıyor
**✅ Düzeltildi:** `ball_detector.py` ve `court_detection_net.py`'deki `infer_model` gövdeleri `with torch.no_grad():` içine alındı.

### 2.3 Batch'lenmemiş, tekrarlı ön işleme — ⏳ açık
- `ball_detector.py`: her adımda 3 kare yeniden resize ediliyor.
- Tüm modeller batch=1 çalışıyor.
- Top ve saha modeli için kare ayrı ayrı resize ediliyor.

**Öneri (değişmedi):** Cache'leme, batch inference, paylaşılan resize.

### 2.4 `sympy` ile kesişim hesabı — aşırı yavaş ve ağır bağımlılık
**✅ Düzeltildi:** `postprocess.py`'deki `line_intersection`, sembolik `sympy.Line.intersection()` yerine basit determinant formülüne (numpy/saf Python) geçirildi. Sympy sürümüne karşı 200 rastgele doğru çifti + paralel doğru edge-case'iyle test edildi: **0 uyuşmazlık**. `sympy` bağımlılığı `requirements.txt`'ten tamamen kaldırıldı.

### 2.5 requirements.txt kurulamaz durumda
**✅ Düzeltildi:** `torch==1.5.0` / `numpy==1.19.3` gibi kurulamayan pinler, projenin venv'inde fiilen doğrulanmış çalışan sürümlere güncellendi: `torch 2.8.0`, `numpy 2.0.2`, `opencv-python 5.0.0.93`, `scipy 1.13.1`, `pandas 2.3.3`, `tqdm 4.68.4`, `catboost 1.2.10`, `matplotlib 3.9.4`, `scenedetect 0.6.7.1`. Temiz bir venv'de `pip install --dry-run -r requirements.txt` ile PyPI çözümlemesi çakışmasız geçti.

### 2.6 Deprecated API'ler
**✅ Düzeltildi (tümü):**
- `F.sigmoid` → `torch.sigmoid` (`court_detection_net.py`, kullanılmayan `F` importu da kaldırıldı).
- `fasterrcnn_resnet50_fpn(pretrained=True)` → `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (`person_detector.py`); gerçek kare üzerinde test edildi, deprecation uyarısı artık çıkmıyor.
- `scenedetect` `VideoManager`/`StatsManager` → modern `open_video`/`SceneManager` (`utils.py`); gerçek `input_video.mp4` üzerinde eski API ile **birebir aynı sahne listesi** üretildiği ve hiçbir deprecation uyarısı çıkmadığı doğrulandı.
- `torch.load(..., weights_only=True)` eklendi (`ball_detector.py`, `court_detection_net.py`).

---

## 3. Orta Öncelikli Bulgular

| Konu | Yer | Detay | Durum |
|---|---|---|---|
| `int(fps)` kaybı | `main.py` | 29.97 fps → 29; uzun videoda ses/süre kayması. `float` tutun. | ⏳ açık |
| Sabit `DIVX` fourcc | `main.py` | `.mp4` çıktı ile uyumsuzluk riski; `mp4v`/`avc1` tercih edin. | ⏳ açık |
| `main()` adı yanıltıcı | `main.py` | Fonksiyon aslında görselleştirme yapıyor; `render_output` gibi bir ad daha uygun. | ⏳ açık |
| CLI doğrulaması yok | `main.py` | Argümanlar `required=True` değildi; dosya varlığı kontrol edilmiyordu. | **✅ Düzeltildi** — tüm path argümanları `required=True`, dosya varlığı `parser.error()` ile kontrol ediliyor. |
| `BallDetector` `path_model=None` tutarsızlığı | `ball_detector.py` | Model yolu verilmezse `.to(device)`/`.eval()` çağrılmıyor. | ⏳ açık |
| Kareler in-place değiştiriliyor | `main.py` | `img_res = frames[i]` kopya değil referans. | ⏳ açık |
| `if len(person[0]) > 0` anlamsız kontrol | `main.py` | Koşul hiçbir şeyi filtrelemiyordu. | **✅ Düzeltildi** — gereksiz koşul kaldırıldı. |
| `refine_kps` eksen karmaşası | `postprocess.py` | `x_ct` aslında satır (y), dönüş değeri ters sırada. | ⏳ açık |
| `detect_lines` kırılganlığı | `postprocess.py` | `HoughLinesP` `None` döndüğünde tesadüfen çalışıyor. | ⏳ açık |
| Hardcoded 12/14 nokta sınırı | `homography.py` | `range(1, 13)` ve `range(12)` sihirli sayılar. | ⏳ açık |
| ConvBlock sırası | `tracknet.py` | `Conv → ReLU → BatchNorm` alışılmadık ama ağırlıklar bu sırayla eğitildi, **değiştirilmemeli**. | Bilinçli olarak değiştirilmedi |

---

## 4. Kod Kalitesi ve Süreç — açık

- **Test yok.** Saf fonksiyonlar (`get_trans_matrix`, `merge_lines`, `line_intersection`, `BounceDetector.postprocess`, `prepare_features`) hâlâ birim testsiz; bu turda yalnızca ad-hoc doğrulama scriptleriyle test edildi, kalıcı test dosyası eklenmedi.
- **Tip ipuçları yok, logging yok.**
- **Konfigürasyon dağınık.** Eşikler kod içine gömülü.
- **Ölü kod:** `court_reference.py`'de kullanılmayan `matplotlib` importu hâlâ duruyor (person_detector testinde bunun tetiklediği pyparsing deprecation uyarıları gözlemlendi).
- **README eksikleri:** Örnek tam komut satırı, ağırlık dosyalarının nereye konacağı, Python sürüm gereksinimi hâlâ belirtilmemiş.

---

## 5. Yeni Özellik: Top Hızı Göstergesi

Rapor kapsamı dışında, ayrı bir istekle **top hızını topun yanına km/h olarak yazan** bir özellik eklendi:
- `speed_estimator.py` (yeni dosya): `get_ball_speed(ball_track, homography_matrices, fps)` — her top noktasını kendi karesinin homografi matrisiyle gerçek saha düzlemine (cm) projelendirip, en fazla 5 kare geriye giderek kat edilen mesafeden km/h hesaplıyor; eksik tespitleri atlıyor, aşırı değerleri (>300 km/h) eliyor, 5 karelik medyan filtreyle titremeyi yumuşatıyor.
- `main.py`: `main()` fonksiyonuna `ball_speed` parametresi eklendi, topun yanına `"XXX km/h"` yazdırılıyor.
- Sentetik testlerle doğrulandı (sabit hareket → doğru km/h, tespit boşluklarında tutarlı sonuç).
- **Bilinen kısıt:** Kamera sabit olmalı ve saha tamamen kadrajda olmalı; homografi kurulamayan karelerde hız gösterilemez. Bu, mevcut mimarinin (§2.1, §2.3) doğal bir sonucu.

---

## 6. Güncel Yol Haritası

**Tamamlanan (bu oturumda):**
- ✅ §1.1–1.5 (tüm kritik bulgular)
- ✅ §2.2, §2.4, §2.5, §2.6 (yüksek öncelik — no_grad, sympy kaldırma, requirements.txt, deprecated API'ler)
- ✅ Orta öncelik: CLI doğrulaması, anlamsız `len(person[0])>0` kontrolü
- ✅ Bonus: top hızı özelliği (§5)

**Kısa vade (1-2 gün) — hâlâ açık:**
1. Streaming pipeline — belleğe tüm videoyu almadan sahne bazlı işle-yaz (§2.1)
2. Batch inference + paylaşılan ön işleme (§2.3)
3. Saf fonksiyonlara kalıcı birim testler (§4)

**Orta vade (1 hafta+) — hâlâ açık:**
4. Kalan orta öncelik maddeleri (§3): `int(fps)`, fourcc, `main()` adlandırması, in-place frame mutasyonu, `refine_kps` netliği, hardcoded sınırlar
5. Konfigürasyon merkezileştirme, logging, tip ipuçları
6. Ölü kod temizliği (`matplotlib` importu vb.)
7. README güncellemesi (tam komut örneği, ağırlık dosyası yerleşimi, Python sürümü)
8. Person detector için daha hızlı alternatif (ör. YOLO ailesi) değerlendirmesi
