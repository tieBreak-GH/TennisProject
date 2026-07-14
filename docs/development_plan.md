# Geliştirme Planı

Bu doküman, `TennisProject`'in tamamlanan işlerini ve önümüzdeki geliştirme fikirlerini tek bir yerde toplar. Kod incelemesi bulguları için [code_review_report.md](code_review_report.md)'ye, mimari açıklaması için [tennis_analysis_report.md](tennis_analysis_report.md)'ye, benzer ürünlerle karşılaştırma ve mobil kullanılabilirlik değerlendirmesi için [comparison_report.md](comparison_report.md)'ye bakın.

## Durum Özeti

Proje şu an: top/saha/oyuncu tespiti (top+saha CNN tabanlı, oyuncu YOLO11n), top hızı ve ral(l)i/servis analizi, minimap görselleştirmesi, CLI ve Streamlit web arayüzü, çoklu GPU backend desteği (CUDA/ROCm/MPS/DirectML) içeriyor. Video işleme, sahne bazlı homografi önbellekleme ve YOLO sayesinde önceki haline göre önemli ölçüde hızlandı (bkz. §7 aşağıda).

## Tamamlanan İşler

- **Kritik/yüksek öncelikli bug düzeltmeleri**: `None`/truthiness hataları, sabit `scale=2` varsayımı, `np.mean([])` uyarısı, `PersonDetector` parametre isimlendirmesi, `torch.no_grad()` eksikliği, `sympy` bağımlılığının kaldırılması, deprecated API güncellemeleri, `requirements.txt` modernizasyonu, CLI argüman doğrulaması (detaylar: [code_review_report.md](code_review_report.md) §1-2).
- **Top hızı özelliği**: Homografi tabanlı km/h hesaplama, topun yanında anlık hız + sabit "vuruş maksimum hızı" HUD'u (`speed_estimator.py`).
- **Web arayüzü**: Streamlit tabanlı `app.py` — video yükleme, oyuncu tespitini atlama seçeneği, sonuç metrikleri, indirme.
- **Çoklu GPU desteği**: CUDA/ROCm otomatik, Apple MPS otomatik, Windows DirectML opsiyonel (`_select_device`, `main.py`).
- **Near real-time performans**: Saha homografisi artık sahne başına önbellekleniyor (~29.7s → ~3.8s); oyuncu tespiti Faster R-CNN'den YOLO11n'e geçirildi (~575ms/kare → ~16ms/kare CPU, ~11ms/kare MPS — performans çukuru yok). Detaylar ve ölçümler: [code_review_report.md](code_review_report.md) §7.
- **Ral(l)i segmentasyonu ve servis tespiti**: Top takibi + sekme verisiyle videoyu ral(l)ilere/vuruşlara bölüp ilk vuruşu baseline yakınlığına göre servis olarak etiketleme (`rally_analyzer.py`), video üzerinde "SERVIS" etiketi, web UI'da ral(l)i tablosu. Detaylar: [code_review_report.md](code_review_report.md) §8.
- **Tek tıkla başlatma script'leri**: macOS/Linux/Windows için `run_app_*` script'leri.
- **README**: Güncel kurulum/çalıştırma talimatları, tüm özelliklerin dokümantasyonu.

## Yol Haritası

### Teknik (mimari / performans) — hâlâ açık
1. **Streaming pipeline**: Tüm video RAM'e yükleniyor (`main.py:read_video`); uzun maçlarda OOM riski. Sahne bazlı oku-işle-yaz mimarisine geçiş gerekiyor.
2. **Batch inference**: Top/saha modelleri hâlâ batch=1 çalışıyor.
3. **`utils.py` çifte video decode**: `scene_detect` (PySceneDetect) videoyu `read_video`'dan bağımsız ikinci kez okuyor. Kazanç küçük görüldüğü için bilinçli olarak ertelendi (bkz. §7).
4. **Birim testler**: Saf fonksiyonlar (`get_trans_matrix`, `line_intersection`, `BounceDetector.postprocess`, `rally_analyzer.py`) hâlâ kalıcı test dosyasız — bu oturumlarda yalnızca ad-hoc scriptlerle doğrulandı.
5. **Konfigürasyon merkezileştirme**: Eşikler (`person_min_score=0.3`, `max_speed_kmh=300`, `max_gap_seconds=1.5`, Hough parametreleri) kod içine gömülü; bir config dosyasına taşınabilir.
6. **ONNX/quantization** (yeni fikir, henüz değerlendirilmedi): CPU-only kullanıcılar için top/saha modellerini ONNX Runtime'a taşımak ekstra hızlanma sağlayabilir.
7. **Video codec: `DIVX` → `avc1` (H.264)** — ⚠️ önceliği yükseltildi (eskiden kozmetik temizlik maddesiydi). Çıktı şu an `FMP4` (MPEG-4 Part 2) olarak yazılıyor; Chrome (masaüstü + Android) bu codec'i HTML5 `<video>` içinde oynatamıyor, yani web arayüzündeki video önizlemesi Chrome kullanıcılarında büyük olasılıkla çalışmıyor (indirme çalışır). `avc1` fourcc'sinin bu makinede OpenCV ile sorunsuz H.264 yazdığı doğrulandı (15 Temmuz 2026); geçiş küçük bir değişiklik, mobil/web kullanımın ön şartı. Detay: [comparison_report.md](comparison_report.md) §5.
8. Kalan orta öncelikli temizlik: `int(fps)` kaybı, `main()` adlandırması, in-place frame mutasyonu, `refine_kps` netliği, ölü kod (`court_reference.py`'deki kullanılmayan `matplotlib` importu).

### Mantık / analiz özellikleri
1. ✅ **Ral(l)i istatistikleri ve servis tespiti** — tamamlandı (bkz. §8 code review raporu).
2. **Oyuncu hareket ısı haritası** — planlanmadı. Zaten hesaplanan oyuncu-minimap noktalarını biriktirip statik bir yoğunluk haritası üretmek; yeni model gerekmez.
3. ✅ **Otomatik highlight üretimi** — tamamlandı. `rally_analyzer.select_highlights` en hızlı vuruşlu ve en uzun ral(l)ileri (kriter başına ayarlanabilir top-N) seçiyor; `main.write_highlights` bu ral(l)ilerin zaten render edilmiş karelerini (top/saha/hız overlay'leriyle, yeniden inference gerekmeden) ayrı kısa mp4 klipler olarak yazıyor. `process_video`'ya `generate_highlights`/`highlights_dir`/`highlights_top_n` parametreleri, `stats`'a `highlight_clips` eklendi; `app.py`'da bir onay kutusu + top-N kaydırıcısı ve her klip için ayrı önizleme/indirme düğmesi var.
4. **İstatistik dışa aktarma (CSV/JSON)** — planlanmadı. `stats['rallies']` zaten yapılandırılmış veri; `app.py`'a bir indirme düğmesi eklemek küçük bir iş.
5. **Forehand/backhand vuruş sınıflandırması** — kullanıcıyla değerlendirildi, şimdilik ertelendi. Poz (pose) tahmini modeli (ör. `yolo11n-pose.pt`) + oyuncu el tercihi (varsayılan/ayarlanabilir) gerektirir; doğruluk garantisi yok, ayrı bir karar/onay gerektirir.
6. **Ölü zaman kırpma ("sadece ral(l)iler" videosu)** — yeni (karşılaştırma raporundan, 15 Tem 2026). SwingVision'ın en sevilen özelliği: maçın ral(l)i olmayan bölümlerini atıp sadece oyun anlarını içeren tek bir video üretmek. Ral(l)i pencereleri (`stats['rallies']`) zaten hesaplandığı için düşük maliyetli — `write_highlights` desenine benzer şekilde pencereleri birleştirip tek dosya yazmak yeterli.
7. **Çizgi kararı (in/out)** — yeni fikir (karşılaştırma raporundan), henüz planlanmadı. Ticari ürünlerin ana özelliği; sekme noktası + homografi + saha çizgileri (CourtReference) zaten mevcut olduğundan teknik temel hazır. Ancak sekme tespitinin kare hassasiyeti ve homografi hatası nedeniyle doğruluk beklentisi düşük tutulmalı; kullanıcıyla kapsam/doğruluk beklentisi netleştirilmeden başlanmamalı.

### UX/UI
1. ✅ **Gerçek ilerleme çubuğu** — tamamlandı. `process_video`'daki `progress_callback` artık `(message, fraction, eta_seconds)` imzasıyla çağrılıyor; `fraction` pipeline aşaması bazında kaba bir ilerleme yüzdesi (per-frame değil), `eta_seconds` geçen süre/`fraction` oranından hesaplanan kalan süre tahmini. `app.py` bunu `st.progress(fraction, text=message)` ile ve altında bir "tahmini kalan süre" alt yazısıyla gösteriyor.
2. **Hız grafiği** — planlanmadı. Mevcut `ball_speed` verisiyle, video yanında zaman-hız çizgi grafiği (ekstra hesaplama gerekmez).
3. **Zaman çizelgesi/scrubber** — planlanmadı. Sekme ve servis anlarının işaretlendiği bir video zaman çizelgesi.
4. **Eşik ayarları paneli** — planlanmadı. Güven eşiği, iz uzunluğu gibi parametreleri arayüzden ayarlanabilir kılmak.
5. ✅ **Toplu video işleme** — tamamlandı. `app.py`'daki `file_uploader` artık `accept_multiple_files=True`; yüklenen videolar sırayla işleniyor (ortak bir ilerleme çubuğu + "Video i/N" göstergesiyle), her video kendi katlanır bölümünde (isim, metrikler, ral(l)i tablosu, video önizleme, indirme düğmesi, varsa highlight klipleri) ayrı ayrı gösteriliyor; bir videodaki hata diğerlerini durdurmuyor.
6. **Mobil erişim ve dağıtım** — yeni (karşılaştırma raporundan, 15 Tem 2026). Arayüzün 375px viewport'ta responsive olduğu ve mobil tarayıcıdan dosya yüklemenin çalıştığı doğrulandı; pratik mobil kullanım için sırasıyla: (a) `avc1` codec düzeltmesi (bkz. Teknik §7 — mobil Chrome'da video önizleme bunun ön şartı), (b) `server.maxUploadSize` artırımı (telefon kamerası kayıtları 200 MB varsayılanını aşabilir), (c) kalıcı sonuç saklama (mobil sekme arka plana alınınca Streamlit oturumu kopabiliyor; işi kuyruklayıp sonucu diske yazmak gerekir), (d) GPU'lu bulut dağıtımı (Docker + VPS). Detaylı değerlendirme: [comparison_report.md](comparison_report.md) §5.

## Önceliklendirme Notu

Bu liste kronolojik bir taahhüt değil, bir menü. Her madde ayrı ayrı tartışılıp plan moduyla detaylandırılmalı — özellikle "forehand/backhand" gibi yeni model gerektiren maddeler için kapsam/risk kullanıcıyla netleştirilmeli.
