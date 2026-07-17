# TennisProject — Detaylı Uygulama Planı

*Hazırlanma tarihi: 16 Temmuz 2026. Bu plan, [mimari_fizik_gpu_degerlendirme.md](mimari_fizik_gpu_degerlendirme.md) raporundaki bulguları koda dökülebilir adımlara çevirir. Sıralama bağımlılığa göredir; her fazın kabul kriteri ve geri-dönüş (fallback) davranışı tanımlıdır.*

## Fazların özeti

| Faz | İş | Bağımlılık | Çaba | Risk |
|:--:|---|---|:--:|:--:|
| **0** | Hazırlık: sentetik test altyapısı + çekim/uyarı notları + küçük düzeltmeler | — | 0.5 gün | Düşük |
| **1** | GPU'yu etkinleştir (DirectML/ROCm) + arayüzde cihaz seçici | — (bağımsız) | 0.5–1 gün | Düşük |
| **2** | Kamera kalibrasyonu: homografiden K (odak) + poz (R, t, C) çıkarımı | Faz 0 | 1–2 gün | Orta |
| **3** | 3B projectile yörünge fit + kamera-yüksekliğinden bağımsız hız | Faz 2 | 2–3 gün | Orta |
| **4** | Entegrasyon: `speed_estimator` dispatcher, fallback, güven skoru, arayüz | Faz 3 | 1–2 gün | Orta |
| **5** | Doğrulama: gerçek maç videosuyla uçtan uca + kamera-yüksekliği değişmezlik testi | Faz 4 | 1 gün | — |

Faz 1 diğerlerinden bağımsızdır, paralel yürütülebilir. Faz 2→3→4→5 zincirlidir. **Ara teslim:** Faz 1 tek başına kullanıcıya hemen değer verir (GPU); Faz 2-5 asıl fizik çözümüdür.

---

## Faz 0 — Hazırlık, küçük düzeltmeler, çekim rehberi

Amaç: sonraki fazların üstüne oturacağı test altyapısını kurmak ve düşük-çabalı, yüksek-etkili düzeltmeleri hemen almak.

### 0.1 Küçük düzeltmeler (her biri bağımsız, ~dakikalar)
- **`main._open_writer` — sessiz başarısızlık kontrolü.** `avc1` encoder yoksa `VideoWriter` exception atmadan boş dosya üretir.
  ```python
  def _open_writer(path, fps, width, height):
      writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
      if not writer.isOpened():
          raise RuntimeError(
              "Video yazıcı açılamadı (avc1/H.264). OpenCV/FFMPEG derlemesinde "
              "H.264 encoder olmayabilir. Yol: {}".format(path))
      return writer
  ```
- **`filter_players` ölü kodunu kaldır** (`person_detector.py`) — hiçbir çağrı noktasında `True` geçilmiyor (bkz. senior raporu §2.2). Fonksiyonu ve ilgili parametreleri sil.
- **README "Video çekim önerileri"ne yükseklik maddesi** (`app.py` içindeki expander + README):
  > - Kamerayı **olabildiğince yüksek ve geride** (baseline arkası, yükseltilmiş) konumlandırın. Alçak kamera (ör. 1.5–2.5 m tripod) top hızını sistematik olarak **abartır**; yüksek kamera hatayı küçültür (bkz. `docs/mimari_fizik_gpu_degerlendirme.md` §3).

### 0.2 Sentetik projeksiyon test altyapısı (Faz 2-3-5'in temeli)
Yeni dosya `tests/synthetic.py` — bilinen bir kamera ve bilinen bir parabol üretip görüntüye yansıtan yardımcılar:
```python
def make_camera(height_cm, pitch_deg, focal_px, image_size, court_center):
    """Bilinen yükseklik/açı/odakta bir kamera → (K, R, t, C, H_court2img)."""
def make_parabola(p0_cm, v0_cms, fps, n_frames, g=981.0):
    """Bilinen p0, v0'dan 3B nokta dizisi (cm) + zaman dizisi."""
def project_points(points_3d, K, R, t):
    """3B cm noktaları → piksel (u, v). Faz 5'in değişmezlik testi bunu kullanır."""
```
Bu, "kod olmadan da matematiği kanıtlayan" referanstır: aynı parabolü farklı `height_cm` kameralara yansıtıp Faz 3'ün hepsinde **aynı hızı** geri kazandığını doğrulayacağız (Faz 5.2).

**Kabul kriteri:** `pytest` yeşil (mevcut 26 + yeni sentetik yardımcı importları); `_open_writer` geçersiz codec'te anlaşılır hata veriyor.

---

## Faz 1 — GPU'yu etkinleştir (RX 9070 XT / AMD, Windows)

Kök neden: kurulu torch `2.8.0+cpu` (CUDA/ROCm yok), `torch_directml` yok. Kod (`main._select_device`) DirectML/ROCm'u zaten otomatik algılıyor ve `process_video(device=...)` parametresi zaten var — **kod değişikliği minimum**, iş çoğunlukla kurulum + arayüz.

### 1.1 Cihaz seçici (arayüz)
`app.py` formuna bir `st.radio` ekle:
```python
device_choice = st.radio('İşlem birimi', ['Otomatik', 'CPU (garantili)', 'GPU (DirectML/ROCm dene)'],
                          help='GPU deneysel backend’lerde sorun çıkarırsa CPU’ya alın.')
```
`process_video`'ya geçir:
```python
device = None                              # Otomatik → _select_device
if device_choice.startswith('CPU'):  device = 'cpu'
elif device_choice.startswith('GPU'): device = None  # _select_device zaten GPU’yu dener
```
`process_video`/`_select_device` imzaları değişmez; sadece `prefer_alt_gpu` ve `'cpu'` zorlaması bu seçime bağlanır. `stats['ball_court_device']` zaten arayüzde gösteriliyor → kullanıcı hangi cihazın seçildiğini görür.

### 1.2 GPU doğrulama betiği
Yeni `check_gpu.py` — bir modeli seçili cihaza yükleyip tek forward pass + zamanlama:
```python
# torch cihazını raporla, DirectML/ROCm/CUDA/MPS dene, 1 kare TrackNet forward süresi yazdır
```
Kullanıcı `python check_gpu.py` ile kartının görünüp görünmediğini ve hızını saniyeler içinde görür.

### 1.3 Kurulum rehberi (README yeni bölüm: "AMD Windows GPU")
- **Yol 1 — DirectML (kolay):** *ayrı* bir venv'de `pip install torch-directml` (torch'u uyumlu eski sürüme düşürür). `check_gpu.py` ile doğrula. RDNA4 yeni olduğundan çıktı doğruluğunu Faz 5 videosuyla teyit et.
- **Yol 2 — WSL2 + ROCm (performanslı):** WSL2 Ubuntu → ROCm PyTorch wheel (`--index-url .../rocmX.Y`); RDNA4 (gfx1201) için ROCm ≥ 6.4.
- **Uyarı:** `requirements.txt`'e GPU torch **konmaz** (platforma göre değişir, CI'yı kırar).

**Kabul kriteri:** `check_gpu.py` DirectML altında RX 9070 XT'yi listeliyor ve CPU'dan hızlı bir forward süresi raporluyor; arayüzden GPU seçilip bir video hatasız işleniyor; GPU yoksa otomatik CPU'ya düşüyor.

> Not: Video **decode/encode** (`cv2.VideoCapture`/`VideoWriter`) FFmpeg ile CPU'da kalır — bu faz onu hızlandırmaz; asıl kazanç model çıkarımındadır (darboğaz orası).

---

## Faz 2 — Kamera kalibrasyonu (homografi → K, R, t, C)

Amaç: mevcut saha homografisinden kameranın **odak uzaklığını** ve **3B pozunu (özellikle yüksekliğini)** kullanıcı girdisi olmadan çıkarmak. Bu, Faz 3'ün 3B ışınlarını kurmasını sağlar.

Yeni modül `camera_calib.py`. Homografi **sahne başına sabit** olduğundan (kod zaten böyle önbelliğe alıyor) poz da **sahne başına bir kez** hesaplanır.

### 2.1 Girdi
`homography_matrices[i]` şu an **image→court** (cm). Court→image gerekli:
```python
H_c2i = np.linalg.inv(H_i2c)   # court(cm) → image(px)
```
`CourtReference` biriminin cm olduğu doğrulandı (baseline–baseline 2374 birim ≈ 2377 cm). Doğrudan cm kullanılır.

### 2.2 Odak uzaklığı (intrinsics)
Sahanın iki dik yön ailesinin kaçış noktaları homografinin ilk iki sütunudur: `h1 = H_c2i[:,0]`, `h2 = H_c2i[:,1]` (X ve Y sonsuz yönlerinin görüntüsü). Kare piksel + sıfır skew + asal nokta = görüntü merkezi varsayımıyla, ω = diag(1/f², 1/f², 1) (merkeze göre) ve dik yön kısıtları f'yi verir:
```
f² = -(a1·b1 + a2·b2) / (a3·b3)                         # h1ᵀ ω h2 = 0  kısıtı
f² = ((a1²+a2²) - (b1²+b2²)) / (b3² - a3²)              # h1ᵀ ω h1 = h2ᵀ ω h2  kısıtı
```
(a=h1, b=h2, asal noktaya göre kaydırılmış koordinatlarda.) İki tahmin birbirini teyit eder; pozitif ve tutarlı olanı seç, değilse EXIF/tipik telefon FOV'undan makul bir `f` ile devam et (fallback).
```python
def estimate_focal(H_c2i, image_size) -> float | None:
    """Saha homografisinden odak uzaklığı (px). Tutarsız/negatif f² → None."""
```

### 2.3 Poz (extrinsics)
```python
def decompose_pose(H_c2i, K):
    B = np.linalg.inv(K) @ H_c2i
    lam = 1.0 / np.linalg.norm(B[:, 0])
    r1, r2 = lam * B[:, 0], lam * B[:, 1]
    r3 = np.cross(r1, r2)
    R = _orthonormalize(np.column_stack([r1, r2, r3]))   # SVD ile en yakın rotasyon
    t = lam * B[:, 2]
    C = -R.T @ t                                          # kamera merkezi (cm)
    return R, t, C          # C[2] = kamera yüksekliği (cm)
```
İşaret belirsizliği (λ ±) topun/kameranın sahanın önünde ve z>0 olması kısıtıyla giderilir (C_z > 0 ve sahne noktaları kameranın önünde).

### 2.4 Testler (`tests/test_camera_calib.py`)
Faz 0.2 sentetik kamerasını kullan: bilinen `height_cm`, `focal_px` → `H_c2i` üret → `estimate_focal` ve `decompose_pose` bunları **geri kazanmalı** (ör. f içinde %2, yükseklik içinde %3). Farklı yükseklik/açıların hepsinde geçmeli.

**Kabul kriteri:** sentetik homografiden f ve kamera yüksekliği tolerans içinde geri kazanılıyor; gerçek maç homografisinde makul bir yükseklik (ör. yayın 8–20 m, tribün) çıkıyor; kötü/dejenere homografide `None` dönüp Faz 4 fallback'ini tetikliyor.

---

## Faz 3 — 3B projectile yörünge fit (kamera-yüksekliğinden bağımsız hız)

Amaç: her uçuş segmentinde (sekme/vuruş arası) topun gerçek 3B yörüngesini geri kazanıp gerçek hızı vermek. **Kamera yüksekliği Faz 2'den geldiği için sonuç yükseklikten bağımsızdır** — kullanıcının sorusunun kalıcı çözümü.

Yeni modül `trajectory_3d.py`.

### 3.1 Piksel → 3B ışın
Kamera merkezi C ortak; her piksel bir yön tanımlar:
```python
def pixel_ray_dir(uv, K, R):
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return d / np.linalg.norm(d)          # dünya koordinatında birim yön
# Topun 3B konumu:  X = C + s·d   (bilinmeyen derinlik s>0)
```

### 3.2 Segment fit — önce lineer, sonra rafinasyon
Bir segmentte N kare: zamanlar t_k = (frame_k − frame_0)/fps, yönler d_k, ortak C. Projectile modeli (cm, s; g=981 cm/s² −z):
```
X_k = p0 + v0·t_k + ½·a·t_k²,   a = (0, 0, −981)
X_k = C + s_k·d_k
```
İkisini eşitle → **(p0, v0, s_k) cinsinden lineer**:
```
p0 + v0·t_k − s_k·d_k = C − ½·a·t_k²      (kare başına 3 denklem)
```
Bilinmeyen: p0(3)+v0(3)+s_k(N) = 6+N; denklem 3N. N≥3 için çözülebilir → `np.linalg.lstsq`. Bu, hem sonucu hem de nonlineer rafinasyonun başlangıcını verir.

**Rafinasyon (opsiyonel ama önerilir):** reprojeksiyon hatasını (piksel) minimize et — istatistiksel olarak doğru amaç:
```python
def _reproj_residuals(params, times, uv_obs, K, R, t):
    p0, v0 = params[:3], params[3:]
    X = p0 + np.outer(times, v0) + 0.5*np.outer(times**2, A_GRAV)
    uv_pred = project(X, K, R, t)          # K[R|t] ile
    return (uv_pred - uv_obs).ravel()
# scipy.optimize.least_squares(_reproj_residuals, x0=lineer_çözüm, method='lm')
```

### 3.3 Segmentten hıza
```python
def segment_speed_series(p0, v0, times):
    v = v0 + np.outer(times, A_GRAV)       # v(t) = v0 + a·t
    return np.linalg.norm(v, axis=1) * 3.6/100   # cm/s → km/h
```
Segmentin tepe hızı genelde başta (vuruş anı) veya sekme öncesidir. Segment başına tek "vuruş hızı" HUD'u (mevcut `get_shot_max_speed` mantığı) bu 3B hız serisinin maksimumundan üretilir.

### 3.4 Fit kalitesi / güven — **uygulanırken ortaya çıkan kritik bulgu**

İlk tasarım `rmse_px` eşiğine dayanıyordu, ancak uygulama sırasında (bkz. `trajectory_3d.py`, gerçekten koda döküldü) ampirik olarak şu bulundu: **`rmse_px` tek başına yanıltıcı.** Topun hareketi kameranın bakış eksenine yakınsa (radyal hareket — ör. kamera baseline arkasındaysa ve top sahayı boyluca geçiyorsa, ki tenis'te en sık görülen vuruş yönü tam olarak budur), piksel gürültüsü reprojeksiyon hatasına değil, ışın-boyunca derinlik/hız belirsizliğine yansıyor. Ölçülen örnek: 1 piksel gürültüyle `rmse_px=0.71` (mükemmel görünüyor) ama gerçek hız **%64 hatalı**. Ayrıca kısa segmentler (parabolün eğriliği gürültü tabanının üstüne henüz çıkmamış — ör. 133 ms) da benzer şekilde `rmse_px`'i düşük tutup hızı büyük ölçüde yanlış verebiliyor (aynı senaryoda 8 karede ortalama %333 hata, 45 karede %0.6'ya düşüyor).

**Çözüm:** `rmse_px` yerine (ya da yanında) fit'in Jacobian'ından **kovaryans tabanlı hız belirsizliği** hesaplanıyor:
```python
residual_var = sum(residuals²) / max(1, 2N - 6)
cov = residual_var * pinv(J.T @ J)                    # parametre kovaryansı (p0, v0)
speed_std_kmh = sqrt(grad @ cov[v0,v0] @ grad) * 0.036 # grad = v0/|v0|, hıza lineerize edilmiş yayılım
```
Bu metrik, hem radyal-hareket hem kısa-segment durumunda gerçek hatayı güvenilir şekilde öngördüğü ampirik olarak doğrulandı (bkz. `tests/test_trajectory_3d.py`). Nihai güven kapısı:
```python
is_reliable_fit = (N >= 5) and (rmse_px <= 8px) and (speed_std_kmh / speed_kmh <= 0.15)
```
`speed_std_kmh/speed_kmh` (bağıl belirsizlik) **birincil**, `N`/`rmse_px` ucuz ikincil sağlık kontrolleri.

**Faz 4 entegrasyonu için sonuç:** Kamera baseline arkasında ve top sahayı boyluca kat ediyorsa (yaygın durum), 3B fit güvenilir olmak için **yeterince uzun bir uçuş segmenti** (ampirik olarak ~500 ms/@60fps eşiği civarı, ama gerçek `speed_std_kmh` oranı üzerinden dinamik olarak belirlenir — sabit bir süre eşiği değil) gerektirir. Kısa segmentlerde (sekmeye çok yakın vuruşlar, hızlı service return'ler) `is_reliable_fit` doğal olarak `False` dönüp 2B fallback'e düşecektir — bu, dispatcher'ın (§4.1) zaten tasarlandığı davranış, ekstra kod gerekmiyor, sadece beklenti şu: **kısa/radyal segmentlerde 3B kazanım daha az olacak, 2B daha sık devrede kalacak.**

### 3.5 Testler (`tests/test_trajectory_3d.py`)
- Bilinen p0, v0 → sentetik parabol → yansıt → fit → **v0'ı %3 içinde** geri kazan.
- Gürültü ekle (±1–2 px), fit'in makul kaldığını doğrula.
- **Değişmezlik testinin çekirdeği (Faz 5.2'de kullanılır):** aynı parabol, farklı kamera yükseklikleri → aynı hız.

**Kabul kriteri:** sentetik segmentte 3B hız tolerans içinde; N<4 veya kötü fit'te temiz fallback.

---

## Faz 4 — Entegrasyon (dispatcher, fallback, arayüz) — ✅ UYGULANDI

`speed_estimator.get_ball_speed_3d` + `estimate_ball_speed` (dispatcher), `main.process_video` bağlaması, `app.py` yöntem rozeti/güven göstergesi ve `tests/test_speed_estimator_3d.py` (11 test, killer test dahil) uygulandı ve tüm test paketi (65 test) yeşil. Ayrıntılar aşağıdaki alt bölümlerde korunmuştur (tasarım referansı olarak).

Amaç: 3B yolu pipeline'a bağlamak; kalibrasyon/fit başarısızsa mevcut 2B yönteme sorunsuz düşmek; kullanıcıya yöntem + güveni göstermek.

### 4.1 `speed_estimator.py` dispatcher
Mevcut `get_ball_speed` (2B) **aynen korunur** (fallback). Yeni:
```python
def get_ball_speed_3d(ball_track, homography_matrices, scenes, fps, bounce_frames):
    """Sahne başına Faz 2 pozu + segment başına Faz 3 fit → 3B hız serisi.
       Poz/fit başarısızsa o segment/sahne için None döndürür (dispatcher 2B’ye düşer)."""

def estimate_ball_speed(ball_track, homography_matrices, scenes, fps, bounce_frames, method='auto'):
    """method='auto': mümkün olan her segmentte 3B, olmayanda 2B.
       '2d'/'3d' ile zorlanabilir. Dönüş: (speeds, method_per_segment)."""
```
`scenes` zaten `analyze_streaming`'den geliyor → poz sahne başına bir kez hesaplanır (verimli).

### 4.2 `main.process_video` bağlama
```python
ball_speed, speed_method = estimate_ball_speed(
    ball_track, homography_matrices, scenes, fps, bounces, method=speed_method_arg)
```
`stats`'a ekle: `'speed_method'` (segment başına '2d'/'3d'), `'camera_height_cm'` (sahne başına, bilgi amaçlı), `'speed_confidence'`.

### 4.3 `max_speed_kmh` sınırının veriyi silmesini düzelt
3B yolda büyütme patlaması olmadığından (Faz 3), 3B segmentlerde sınır **gevşetilir/kaldırılır**; sadece 2B fallback segmentlerinde eski koruma kalır. Böylece gerçek hızlı vuruşlar artık sessizce `None`'a düşmez.

### 4.4 Arayüz (`app.py`)
- Metriklerin yanında yöntem rozeti: "Hız: 3B kalibre (kamera ≈ H m)" ya da "2B tahmini (düşük güven)".
- Hız grafiğinde 3B/2B segmentlerini ayırt eden bir not.
- `st.info`: 3B başarısızsa nedenini açıkla (saha görünmüyor / az kare / dejenere homografi).

**Kabul kriteri:** saha net görünen videoda 3B devreye giriyor; saha yok/kötüyse 2B'ye düşüp çökmüyor; her iki yol da geçerli çıktı veriyor; mevcut 26 test hâlâ yeşil.

---

## Faz 5 — Uçtan uca doğrulama

### 5.2 Kamera-yüksekliği değişmezlik testi — ✅ UYGULANDI (dispatcher seviyesinde)

`tests/test_speed_estimator_3d.py::test_dispatcher_3d_speed_is_invariant_to_camera_height` (200/500/1200 cm, parametrize) ve
`test_legacy_2d_method_diverges_across_camera_heights_that_dispatcher_does_not` gerçek `estimate_ball_speed` dispatcher'ı üzerinden çalışıyor (izole `trajectory_3d` testinin ötesinde). Ölçülen: aynı atış için eski 2B yöntem 200cm'de kendi >300km/h aykırı-değer korumasıyla tamamen elenirken, 500cm'de 203 km/h, 1200cm'de 115 km/h veriyor (gerçek ~81 km/h); yeni dispatcher'ın 3B sonucu üç yükseklikte de ~81 km/h'de kalıyor (<%3 varyans). Bu, orijinal "farklı kamera yüksekliğinde doğru top hızı" sorusuna doğrudan, regresyona karşı korumalı bir yanıttır (sentetik veriyle).

### 5.1 Gerçek maç videosu — ✅ UYGULANDI — **kritik bulgu: gerçek yayın kamerasında 3B yöntem devreye giremiyor**

Kullanıcının sağladığı gerçek klip (Federer–Nadal, Avustralya Açık 2017, 5. set, 1138 kare, 1280x720, ~45s, saha net görünüyor) ile `main.process_video` uçtan uca çalıştırıldı (CPU, ~11 dakika). Video çökmeden tamamlandı, çıktı videosu (top izi, hız etiketi, minimap, sabit HUD, ral(l)i tablosu) doğru render edildi.

**Ancak ilk çalıştırmada `max_speed_kmh=930 km/h`, `avg_speed_kmh=142 km/h` çıktı** — fiziksel olarak imkânsız. Kök neden analizi (`analyze_cache.pkl` üzerinden, modelleri tekrar çalıştırmadan):
- Bu video, klasik **uzun odaklı (f≈2506px), sahanın arkasından, yükseltilmiş (~9.3 m) yayın kamerası** açısını kullanıyor — yani tam olarak Faz 3.4'ün uyardığı **radyal hareket** durumu: sahayı boyluca kat eden bir ralinin neredeyse tüm vuruşları kameranın bakış eksenine yakın hareket ediyor.
- **19 sekme-arası segmentin TAMAMI** fiziksel olarak imkânsız bir 3B çözüme yakınsadı (300 km/h ile 3×10⁸ km/h arası; `p0`'ın ima ettiği konum bazen sahadan **kilometrelerce** uzakta/altında).
- Bunlardan **3 tanesi** `is_reliable_fit`'in kovaryans-tabanlı eşiğini geçti (`speed_std_kmh/speed` oranı ~%0.04 gibi yanıltıcı derecede düşüktü) — yani mevcut güvenilirlik testi bu durumu **yakalayamadı**. Sebep: kovaryans tahmini yalnızca scipy'nin LM çözücüsünün yakınsadığı yerel optimumun civarındaki eğriliği yansıtıyor; problem küresel olarak kötü-koşullandırılmış (birbirinden çok farklı ama benzer reprojeksiyon hatası veren çözümler var) olduğunda bunu göremiyor.

**Uygulanan düzeltme:** `trajectory_3d.is_reliable_fit`'e sabit bir fiziksel olabilirlik tavanı eklendi (`max_speed_kmh=300.0`, varsayılan) — kovaryans oranı ne kadar "güvenli" görünürse görünsün, 300 km/h üstü hiçbir gerçek tenis vuruşu yok. Bu tek kontrol, gözlemlenen tüm imkânsız sonuçları eledi. Düzeltmeden sonra bu videoda **3B yöntem 0 karede devreye giriyor** (tamamı 2B'ye düşüyor), sonuç: `max_speed_kmh=210 km/h`, `avg_speed_kmh=83 km/h`, `min=0` — fiziksel olarak makul (bkz. `tests/test_trajectory_3d.py::test_is_reliable_fit_rejects_physically_impossible_speed_despite_tiny_uncertainty`, gerçek videodan alınan bu fit ile regresyon testi olarak eklendi).

**Önemli, dürüst sonuç:** 3B yöntem kendini imkânsız çıktı üretmekten korumayı başardı (güvenli davranış), ama **bu spesifik kamera açısı için orijinal sorunu (yükseklikten bağımsız hız) çözemedi** — sistem bu videoda tamamen eski 2B yönteme düşüyor ve o yöntemin yükseklik-bağımlı hatası bu görüntüde düzeltilmeden kalıyor. 3B yöntemin pratik faydası, kameranın topun hareketine göre daha fazla paralaks gördüğü açılarda (yandan/köşeden, daha alçak/yakın geniş-açı) daha yüksek; sahanın tam arkasından uzun odaklı klasik yayın açısı - en yaygın TV kurulumu - şu anki haliyle neredeyse her zaman 2B'ye düşecektir. Bu, gelecekteki bir iyileştirme için açık bir yol haritası bulgusu: tek-karesel ışın+yerçekimi modeli yerine (a) çoklu-vuruş üzerinden fiziksel bir önsel (p0'ın saha sınırları içinde olması gibi) eklemek, veya (b) yalnızca yeterli yanal paralaksı olan sahne/kameralarda 3B'yi devreye sokmak gerekebilir.

**Kabul kriteri (revize):** 5.2'de 3B hızlar yükseklikten bağımsız (varyans < %3, sentetik) — ✅. 5.1'de gerçek video çökmeden tamamlanıyor ve **nihai** hızlar fiziksel olarak makul — ✅ (düzeltmeden sonra); ancak bu spesifik videoda 3B yöntem hiç devreye girmiyor, bu yüzden kamera-yüksekliğinden bağımsızlık faydası bu görüntüde henüz kanıtlanamadı — sadece sentetik testlerde kanıtlandı.

---

## Riskler ve azaltımlar

| Risk | Azaltım |
|---|---|
| Tek görüntüden odak (f) tahmini gürültülü olabilir | İki kısıttan çapraz doğrula; tutarsızsa EXIF/tipik FOV fallback; en kötü halde 2B'ye düş |
| Segment çok kısa (N<4) → fit belirsiz | Eşikle 2B fallback + "düşük güven" etiketi |
| Homografi kalitesi düşük (kötü keypoint) | Zaten `get_trans_matrix` reprojeksiyon hatasıyla en iyisini seçiyor; poz reprojeksiyon RMSE'siyle ikinci bir kalite kapısı |
| RDNA4 DirectML/ROCm olgunlaşmamış | Faz 1'de doğrulama betiği + otomatik CPU fallback; GPU zorunlu değil |
| 3B karmaşıklığı mevcut davranışı bozar | 2B tamamen korunur; 3B additive ve `method='2d'` ile kapatılabilir |

## Bağımlılık notu
Yeni Python bağımlılığı **gerekmez** — `scipy` (`optimize.least_squares`) ve `numpy` zaten kurulu. GPU tarafı (`torch-directml`) opsiyonel ve `requirements.txt` dışında tutulur.

## Önerilen başlangıç
En yüksek getirili ilk adım **Faz 1 (GPU) + Faz 0 küçük düzeltmeler** — birkaç saatte kullanıcıya somut değer. Ardından fizik zinciri **Faz 2 → 3 → 4 → 5**. Faz 2 ve 3 saf matematik + saf-fonksiyon olduğu için sentetik testlerle (model/GPU gerekmeden) tam doğrulanabilir; bu, riski en aza indiren sıralamadır.
