# TennisProject — Senior Kod İncelemesi Raporu

**Tarih:** 14 Temmuz 2026
**Kapsam:** Proje kök dizinindeki tüm Python kaynak dosyaları (venv/weights hariç)
**Genel değerlendirme:** Araştırma/prototip kalitesinde, çalışan bir bilgisayarlı görü pipeline'ı. Mimari kurgu (sahne tespiti → top takibi → saha tespiti → oyuncu tespiti → sekme tespiti → görselleştirme) doğru düşünülmüş. Ancak üretim kalitesine taşınması için düzeltilmesi gereken **gerçek hata riski taşıyan noktalar**, ciddi **bellek/performans sorunları** ve **bakım kalitesi eksikleri** var.

---

## 1. Kritik Bulgular (çökme veya sessiz yanlış sonuç riski)

### 1.1 Sekme (bounce) karesinde top koordinatı `None` ise program çöker
`main.py:95-98` — `if i in bounces and inv_mat is not None` koşulunda `ball_track[i]`'nin geçerli olduğu kontrol edilmiyor. `BounceDetector.smooth_predictions` interpolasyonu `main.py:164-165`'te oluşturulan **kopya listeler** üzerinde çalışır; `main()`'e geçirilen orijinal `ball_track` güncellenmez. Yani sekme tespit edilen karede `ball_track[i] = (None, None)` olabilir ve `np.array((None, None), dtype=np.float32)` `TypeError` fırlatır.

**Öneri:** Ya `predict()`'ten dönen yumuşatılmış koordinatları görselleştirmede de kullanın, ya da çizimden önce `ball_track[i][0] is not None` kontrolü ekleyin.

### 1.2 Sabit `scale=2` — 1280x720 dışındaki videolarda sessizce yanlış koordinat
`ball_detector.py:44` ve `court_detection_net.py:22` — model çıktısı 640x360'tan orijinal çözünürlüğe sabit 2 katsayısıyla ölçekleniyor. README "1280x720 video hazırlayın" diyor ama kod bunu **doğrulamıyor**. 1920x1080 bir girişte program hatasız çalışır fakat top ve saha noktaları tamamen yanlış yere çizilir.

**Öneri:** `scale_x = frame_width / 640`, `scale_y = frame_height / 360` şeklinde dinamik hesaplayın; veya girişte çözünürlük doğrulaması yapıp anlaşılır bir hata verin.

### 1.3 Truthiness ile `None` kontrolü: `x == 0` koordinatı kaybolur
Yaygın kalıp: `if ball_track[i][0]:` (`main.py:67`), `if prev_pred[0]:` (`ball_detector.py:62`), `if not x_ball[num]:` (`bounce_detector.py:66`). Koordinat `0` veya `0.0` olduğunda bu kontroller `None` ile aynı davranır. Kenar durumu ama sessiz veri kaybıdır ve `is None / is not None` maliyetsiz düzeltmedir.

### 1.4 `np.mean([])` → `nan` (homography.py:35)
Bir konfigürasyonun 4 noktası bulunmuş ama geriye kalan hiçbir nokta tespit edilememişse `dists` boş kalır; `np.mean([])` `RuntimeWarning` ile `nan` döner. `nan < dist_max` her zaman `False` olduğundan geçerli olabilecek tek matris de elenir. Ayrıca değişken adları yanıltıcı: `dist_median` aslında ortalama, `dist_max` aslında takip edilen minimum.

**Öneri:** `if not dists: continue` (veya 4 noktalık eşleşmeyi kabul et) + isimleri `mean_error`, `best_error` gibi düzeltin.

### 1.5 `PersonDetector(device)` — `dtype` parametresine device string geçiliyor
`person_detector.py:11` imzası `def __init__(self, dtype=torch.FloatTensor)`, fakat `main.py:159`'da `PersonDetector(device)` çağrılıyor. `'cuda'` stringi `nn.Module.to()` tarafından device olarak kabul edildiği için **tesadüfen** çalışıyor. Parametre `device: str = 'cpu'` olarak yeniden adlandırılmalı; mevcut hali ilk bakışta tip hatası gibi görünüyor ve default değer (`torch.FloatTensor` sınıfı) `.to()`'ya geçirilse zaten geçersiz.

---

## 2. Yüksek Öncelikli Bulgular

### 2.1 Tüm video RAM'e yükleniyor
`main.py:12-23` — `read_video` her kareyi listede tutuyor. 720p, 30fps, 5 dakikalık video ≈ 9000 kare × ~2.7 MB ≈ **24 GB** ham bellek. Ayrıca sonuç kareleri de (`imgs_res`) ikinci bir liste olarak birikiyor. Kısa kliplerde çalışır, gerçek maç videosunda OOM kaçınılmaz.

**Öneri:** Pipeline'ı akış (streaming) mimarisine geçirin: kareleri sahne sahne oku-işle-yaz; ya da en azından çıktıyı biriktirmek yerine anında `VideoWriter`'a yazın (çizim zaten kareleri in-place değiştiriyor, `imgs_res` listesi büyük ölçüde gereksiz).

### 2.2 `torch.no_grad()` kullanılmıyor
`ball_detector.py:37` ve `court_detection_net.py:32` — inference sırasında autograd grafiği kuruluyor; bellek kullanımı ve süre gereksiz artıyor (`person_detector.py` doğru şekilde `no_grad` kullanıyor). İki `infer_model` gövdesini `with torch.no_grad():` (veya `torch.inference_mode()`) içine alın.

### 2.3 Batch'lenmemiş, tekrarlı ön işleme
- `ball_detector.py:29-31`: her adımda 3 kare yeniden resize ediliyor; önceki iterasyonun sonuçları cache'lense resize maliyeti 3'te 1'e iner.
- Tüm modeller batch=1 çalışıyor; GPU'da 8-16'lık batch'ler top/saha tespitini birkaç kat hızlandırır.
- Aynı kare hem top hem saha modeli için ayrı ayrı 640x360'a resize ediliyor; tek resize paylaşılabilir.

### 2.4 `sympy` ile kesişim hesabı — aşırı yavaş ve ağır bağımlılık
`postprocess.py:8-20` — iki doğrunun kesişimi için sembolik matematik kütüphanesi kullanılıyor. Bu, kare başına 11 keypoint × sembolik çözüm demek; basit determinant formülü (birkaç satır numpy) yüzlerce kat hızlıdır ve `sympy` bağımlılığını tamamen kaldırır.

### 2.5 requirements.txt kurulamaz durumda
`torch==1.5.0`, `numpy==1.19.3`, `matplotlib==3.1.2`, `opencv_python==4.1.2.30` pinleri modern Python (3.10+) ve Apple Silicon'da derlenemez/bulunamaz. Proje şu an fiilen "temiz kurulumdan çalıştırılamaz" durumda.

**Öneri:** Güncel sürümlerle yeniden pinleyin (torch ≥ 2.x, numpy ≥ 1.26, scenedetect ≥ 0.6.3...), Python sürümünü belirtin. `matplotlib` yalnızca `court_reference.py`'de kullanılmayan bir import — bağımlılıktan tamamen çıkarılabilir.

### 2.6 Deprecated API'ler
- `F.sigmoid` (`court_detection_net.py:33`) → `torch.sigmoid`.
- `fasterrcnn_resnet50_fpn(pretrained=True)` (`person_detector.py:12`) → `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT`.
- `scenedetect` `VideoManager` (`utils.py`) → 0.6.x'te `open_video` + `SceneManager` önerilen yol.
- `torch.load` çağrılarına `weights_only=True` eklenmeli (güvenlik + gelecek sürüm uyumu).

---

## 3. Orta Öncelikli Bulgular

| Konu | Yer | Detay |
|---|---|---|
| `int(fps)` kaybı | `main.py:14` | 29.97 fps → 29; uzun videoda ses/süre kayması. `float` tutun. |
| Sabit `DIVX` fourcc | `main.py:129` | `.mp4` çıktı ile uyumsuzluk riski; `mp4v`/`avc1` tercih edin veya uzantıya göre seçin. |
| `main()` adı yanıltıcı | `main.py:32` | Fonksiyon aslında görselleştirme yapıyor; `render_output` gibi bir ad, gerçek akış da `if __name__` bloğundan bir `run()` fonksiyonuna taşınmalı. |
| CLI doğrulaması yok | `main.py:138-144` | Argümanlar `required=True` değil; eksik path'te `cv2` sessizce boş video okur. Dosya varlığı da kontrol edilmeli. |
| `BallDetector` `path_model=None` tutarsızlığı | `ball_detector.py:12-15` | Model yolu verilmezse `.to(device)` ve `.eval()` hiç çağrılmıyor; rastgele ağırlıklı model CPU'da kalıyor. Yol zorunluysa parametre opsiyonel olmamalı. |
| Kareler in-place değiştiriliyor | `main.py:63-121` | `img_res = frames[i]` kopya değil referans; orijinal kare listesi kalıcı olarak boyanıyor. Ya bilinçli olduğunu belirtin ya `frames[i].copy()` kullanın. |
| `if len(person[0]) > 0` anlamsız kontrol | `main.py:107` | `person[0]` her zaman 4 elemanlı bbox; koşul hiçbir şeyi filtrelemiyor. |
| `refine_kps` eksen karmaşası | `postprocess.py:22-45` | `x_ct` aslında satır (y), dönüş değeri ters sırada. Çalışıyor ama okuyanı yanıltıyor; `row/col` adlandırması netleştirir. |
| `detect_lines` kırılganlığı | `postprocess.py:51` | `HoughLinesP` `None` döndüğünde `np.squeeze(None)` tesadüfen 0-d array üretip çalışıyor. Açık `if lines is None: return []` yazın. |
| Hardcoded 12/14 nokta sınırı | `homography.py:23,32` | `range(1, 13)` ve `range(12)` sihirli sayılar; `court_conf` uzunluğundan türetilmeli. |
| ConvBlock sırası | `tracknet.py:7-11` | `Conv → ReLU → BatchNorm` alışılmadık (standart: Conv→BN→ReLU). Ağırlıklar bu sırayla eğitildiği için **değiştirmeyin**, ama yeni eğitimde düzeltilmeye değer. |

---

## 4. Kod Kalitesi ve Süreç

- **Test yok.** En azından saf fonksiyonlar (`get_trans_matrix`, `merge_lines`, `BounceDetector.postprocess`, `prepare_features`) hızlıca birim teste alınabilir — bunlar model gerektirmez.
- **Tip ipuçları yok, logging yok.** `print('ball detection')` yerine `logging`; fonksiyon imzalarına tip ekleyin (özellikle `Optional` dönen `postprocess` gibi yerlerde hata sınıfını görünür kılar).
- **Konfigürasyon dağınık.** Eşikler (`0.45`, `max_dist=80`, `person_min_score=0.85`, minimap boyutları) kod içine gömülü; bir `config.py`/dataclass altında toplanmalı.
- **Ölü kod:** `court_reference.py`'de kullanılmayan `matplotlib` importu ve yorum satırı halinde bırakılmış eski kod blokları; `postprocess.py:33`'te yorumlanmış `print`.
- **Git hijyeni:** Son commit mesajı ("initialize virtual environment dependencies...") gerçek içeriğiyle (.gitignore + homography düzeltmesi) uyuşmuyor. `.gitignore`'un venv/weights/video'ları dışlaması doğru karar olmuş.
- **README eksikleri:** Örnek tam komut satırı (`python main.py --path_ball_track_model ...`), ağırlık dosyalarının nereye konacağı ve Python sürüm gereksinimi belirtilmemiş.

---

## 5. Önerilen Yol Haritası

**Hemen (1-2 saat):**
1. Bounce çiziminde `None` koordinat guard'ı (§1.1)
2. `torch.no_grad()` eklenmesi (§2.2)
3. `is not None` düzeltmeleri (§1.3), `np.mean([])` guard'ı (§1.4)
4. CLI `required=True` + dosya varlık kontrolü

**Kısa vade (1-2 gün):**
5. Dinamik scale hesabı veya çözünürlük doğrulaması (§1.2)
6. requirements.txt modernizasyonu + kurulum testi (§2.5)
7. `sympy` → numpy kesişim formülü (§2.4)
8. Deprecated API geçişleri (§2.6)
9. Saf fonksiyonlara birim testler

**Orta vade (1 hafta+):**
10. Streaming pipeline — belleğe tüm videoyu almadan sahne bazlı işle-yaz (§2.1)
11. Batch inference + paylaşılan ön işleme (§2.3)
12. Konfigürasyon merkezileştirme, logging, tip ipuçları
13. Person detector için daha hızlı alternatif (ör. YOLO ailesi) değerlendirmesi
