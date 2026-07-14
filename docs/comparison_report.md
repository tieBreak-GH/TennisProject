# TennisProject — Benzer Ürünlerle Karşılaştırma ve Değerlendirme Raporu

**Tarih:** 15 Temmuz 2026
**Kapsam:** Projenin açık kaynak ve ticari alternatiflerle karşılaştırılması, güçlü/zayıf yön değerlendirmesi ve mobil (web üzerinden) kullanılabilirlik analizi. Mimari detaylar için [tennis_analysis_report.md](tennis_analysis_report.md), geliştirme geçmişi için [code_review_report.md](code_review_report.md), yol haritası için [development_plan.md](development_plan.md).

---

## 1. Özet

Bu proje, [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) açık kaynak projesinin üzerine inşa edilmiş, önemli ölçüde genişletilmiş bir türevidir. Upstream proje yalnızca top/saha/oyuncu/sekme **tespiti** yapan bir araştırma prototipi iken, bu çatal (fork) şunları eklemiştir: top hızı hesabı (km/h), ral(l)i/vuruş segmentasyonu, servis tespiti, otomatik highlight klipleri, Streamlit web arayüzü (ilerleme çubuğu, toplu yükleme), çoklu GPU backend desteği ve ~15x'lik uçtan uca performans iyileştirmesi.

Konumlandırma olarak proje, **ücretsiz/self-hosted açık kaynak araçlar ile ticari tüketici ürünleri (SwingVision vb.) arasındaki boşlukta** duruyor: açık kaynak alternatiflerin hepsinden daha fazla analiz özelliği sunuyor, ama ticari ürünlerin gerçek zamanlı takip, çizgi kararı (line calling) ve mobil uygulama deneyimine sahip değil.

---

## 2. Açık Kaynak Alternatiflerle Karşılaştırma

| Özellik | **Bu proje** | [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) (upstream) | [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) | [anushacodes/tennis-analysis-with-cv](https://github.com/anushacodes/tennis-analysis-with-cv) |
|---|---|---|---|---|
| Top tespiti | TrackNet | TrackNet | TrackNet | TrackNet |
| Saha tespiti | CNN (14 keypoint) + sahne bazlı önbellek | CNN (14 keypoint), her karede | Klasik CV (çizgi tespiti) | ResNet50 keypoint |
| Oyuncu tespiti | YOLO11n (~11-16 ms/kare) | Faster R-CNN (~575 ms/kare CPU) | ResNet50 tabanlı | YOLOv8 |
| Sekme tespiti | CatBoost | CatBoost | — | — |
| Top hızı (km/h) | ✅ homografi ile gerçek düzlemde | ❌ | ❌ | ✅ |
| Ral(l)i/vuruş segmentasyonu | ✅ | ❌ | ❌ | ❌ |
| Servis tespiti | ✅ (baseline yakınlığı kuralı) | ❌ | ❌ | ❌ |
| Otomatik highlight | ✅ (en hızlı/en uzun ral(l)i) | ❌ | ❌ | ❌ |
| Web arayüzü | ✅ Streamlit (ilerleme, toplu yükleme) | ❌ (yalnız CLI) | ❌ | ❌ |
| GPU desteği | CUDA/ROCm/MPS/DirectML | CUDA | CUDA | CUDA |
| Minimap | ✅ | ✅ | ✅ | ✅ |

**Sonuç:** Açık kaynak alanda bu proje, bilinen alternatiflerin en kapsamlısı. Upstream'e göre eklenen analiz katmanı (hız → vuruş → ral(l)i → servis → highlight) diğer projelerin hiçbirinde bütün olarak yok.

---

## 3. Ticari Ürünlerle Karşılaştırma

| | **Bu proje** | SwingVision | Baseline Vision | PlaySight SmartCourt |
|---|---|---|---|---|
| Model | Ücretsiz, açık kaynak, self-hosted | $179.99/yıl abonelik | $2.000 tek seferlik donanım | Kurumsal (saha kurulumu, özel fiyat) |
| Platform | Web (Streamlit), CLI | iOS/iPadOS/Apple Watch uygulaması | Taşınabilir kamera cihazı | Sahaya monte çoklu kamera |
| İşleme | Sonradan (video yükle → işle) | Gerçek zamanlı (cihaz üstünde) | Gerçek zamanlı | Gerçek zamanlı + canlı yayın |
| Top hızı | ✅ (kamera sabit + saha kadrajda şartıyla) | ✅ (±%10 doğruluk iddiası) | ✅ | ✅ |
| Çizgi kararı (in/out) | ❌ | ✅ (%97 iddiası, yakın toplarda) | ✅ (ana özelliği) | ✅ |
| Vuruş türü sınıflandırma | ❌ (planda ertelendi) | ✅ (forehand/backhand/servis vb.) | Kısmi | ✅ |
| Otomatik highlight / ölü zaman kırpma | ✅ highlight; ölü zaman kırpma planda | ✅ ("2 saatlik maçı 15 dakikada izle") | Kısmi | ✅ |
| Skor takibi | ❌ | ✅ | ✅ | ✅ |
| Veri sahipliği/gizlilik | ✅ tamamen yerel | Bulut | Cihaz + uygulama | Bulut |

**Sonuç:** Ticari ürünlerin ortak güçlü yanları gerçek zamanlı işleme, çizgi kararı ve cilalı mobil deneyim. Bu projenin ayırt edici avantajları ise **ücretsiz olması, verinin tamamen yerelde kalması ve kodun değiştirilebilir olması**. SwingVision'ın en sevilen özelliği olan "ölü zamanı at, sadece ral(l)ileri izle" fikri, bu projede ral(l)i segmentasyonu zaten var olduğu için **düşük maliyetle eklenebilir durumda** (geliştirme planına eklendi).

---

## 4. Genel Değerlendirme

**Güçlü yönler**
- Açık kaynak alternatifler arasında en geniş özellik seti; upstream'in atıl bıraktığı tespit çıktılarının (iz, sekme, homografi) üzerine gerçek analiz katmanı kurulmuş.
- Near real-time performans (713 karelik klip ~56 s, oyuncu tespiti dahil) — bu sınıftaki self-hosted araçlar için iyi bir seviye.
- Donanım esnekliği: CUDA/ROCm/MPS (DirectML doğrulanmamış) — rakip açık kaynak projeler pratikte CUDA'ya bağımlı.
- Web arayüzü sayesinde teknik olmayan kullanıcıya da hitap ediyor (CLI zorunlu değil).

**Zayıf yönler / riskler**
- Sabit kamera + sahanın tamamen kadrajda olması şartı: telefonla elde çekilmiş videolarda sonuç garantisi yok. Ticari ürünler bu kısıtı ya donanımla (Baseline, PlaySight) ya da cihaz-üstü gerçek zamanlı takiple (SwingVision) çözüyor.
- Kalibre edilmemiş eşikler (`max_gap_seconds`, `baseline_margin_cm`): servis/ral(l)i tespiti gerçek ve çeşitli videolarla doğrulanmadı; pozitif servis senaryosu hâlâ test edilemedi.
- Çizgi kararı ve skor takibi yok — ticari ürünlerin ana satış noktaları.
- Tüm video RAM'e yükleniyor: uzun maç kayıtları (ticari ürünlerin ana kullanım senaryosu) bu mimariyle işlenemez. Streaming pipeline (planda §Teknik 1) bu yüzden ticari kullanılabilirlik açısından da kritik.
- Kalıcı birim test yok; hukuki/etik değil ama sürdürülebilirlik riski.

---

## 5. Mobil (Web Üzerinden) Kullanılabilirlik Değerlendirmesi

375x812 (iPhone) viewport'ta canlı test edildi. Bulgular:

**Çalışanlar ✅**
- Streamlit arayüzü responsive: başlık, yükleme alanı, onay kutuları, slider ve butonlar mobil genişlikte düzgün diziliyor, dokunma hedefleri yeterli boyutta.
- Dosya yükleme mobil tarayıcıda çalışır: "Browse files" telefonun galeri/kamera seçicisini açar; kullanıcı kortta çektiği videoyu doğrudan yükleyebilir.
- İlerleme çubuğu + ETA ve sonuç ekranı (metrikler, tablo, video, indirme) mobilde de kullanılabilir.

**Engeller / riskler ⚠️**
1. **Erişim:** Sunucu şu an localhost'ta. Aynı Wi-Fi ağındaki telefondan `http://<makine-IP>:8501` (Streamlit'in "Network URL"i) ile hemen kullanılabilir; internet üzerinden kullanım için bulut dağıtımı (Docker + bir VPS veya Streamlit Community Cloud — ancak model ağırlıkları ve GPU ihtiyacı nedeniyle GPU'lu bir sunucu gerekir) şart.
2. **Video codec (kritik, doğrulandı):** Çıktı videosu `DIVX`→`FMP4` (MPEG-4 Part 2) ile yazılıyor. Chrome (masaüstü ve Android) bu codec'i HTML5 `<video>` içinde **oynatamaz** — `st.video` bileşeni mobil Chrome'da büyük olasılıkla siyah ekran gösterir (indirme yine çalışır). Bu makinede `avc1` (H.264) fourcc'sinin OpenCV ile sorunsuz yazıldığı test edilip doğrulandı; `DIVX` → `avc1` geçişi hem masaüstü hem mobil tarayıcı oynatmayı düzeltir. Geliştirme planındaki mevcut "sabit DIVX fourcc" maddesi bu bulguyla **kozmetikten kritik önceliğe** yükseltildi.
3. **Uzun işlem + mobil sekme askıya alma:** İşleme dakikalar sürüyor; mobil tarayıcı arka plana alınan sekmenin WebSocket bağlantısını koparabilir ve Streamlit oturumu (sonuçlarla birlikte) kaybolabilir. Kısa kliplerde sorun düşük, uzun videolarda gerçek risk. Kalıcı çözüm: işleri sunucu tarafında kuyruklayıp sonucu diske yazmak ve sayfa yenilendiğinde geri sunmak.
4. **Yükleme limiti:** Streamlit varsayılanı 200 MB/dosya; telefon kamerasının 1-2 dakikalık 1080p kaydı bunu aşabilir. `server.maxUploadSize` ayarıyla yükseltilebilir (config değişikliği yeterli).
5. **İşlem telefonda değil sunucuda:** Bu bir mimari gerçek — telefon yalnızca ince istemci. SwingVision gibi cihaz-üstü işleme bu mimaride mümkün değil ve hedeflenmemeli.

**Karar:** Evet, web üzerinden mobilde **kullanılabilir** — arayüz mobil uyumlu ve aynı ağda hemen çalışır. Pratik kullanım için sırasıyla: (a) `avc1` codec düzeltmesi (küçük iş, kritik etki), (b) `maxUploadSize` artırımı (config), (c) kalıcı sonuç saklama (orta iş), (d) GPU'lu bulut dağıtımı (büyük iş) gerekiyor. Bu maddeler geliştirme planına eklendi.

---

## 6. Geliştirme Planına Yansıyan Değişiklikler

Bu karşılaştırmanın sonucunda [development_plan.md](development_plan.md)'de yapılan güncellemeler:
1. **"Sabit DIVX fourcc" maddesi kritik önceliğe yükseltildi** — mobil/web oynatma engeli olduğu doğrulandı; `avc1` çözümü bu makinede test edildi.
2. **Yeni UX/UI maddesi: Mobil erişim ve dağıtım** — LAN erişimi, upload limiti, oturum kalıcılığı, bulut dağıtımı alt adımlarıyla.
3. **Yeni Mantık maddesi: Ölü zaman kırpma ("sadece ral(l)iler" videosu)** — SwingVision'ın en sevilen özelliği; ral(l)i segmentasyonu hazır olduğundan düşük maliyetli.
4. **Yeni Mantık fikri: Çizgi kararı (in/out)** — ticari ürünlerin ana özelliği; sekme noktası + homografi zaten mevcut olduğundan teknik temel hazır, ancak doğruluk beklentisi netleştirilmeden başlanmamalı (not olarak eklendi).

---

## Kaynaklar

- [SwingVision resmi site](https://swing.vision/home/) · [App Store sayfası](https://apps.apple.com/us/app/swingvision-tennis-pickleball/id989461317) · [Tennisnerd incelemesi](https://www.tennisnerd.net/tennis-tools/swingvision-review-and-interview/25702) · [TechInTheSun incelemesi](https://techinthesun.com/swingvision/)
- [Baseline Vision](https://www.baselinevision.com/) · [ürün sayfası](https://www.baselinevision.com/product) · [TennisLeo incelemesi](https://www.tennisleo.com/baseline-vision-review/)
- [PlaySight Tennis SmartCourt](https://playsight.com/our-sports/tennis/)
- [yastrebksv/TennisProject (upstream)](https://github.com/yastrebksv/TennisProject) · [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) · [anushacodes/tennis-analysis-with-cv](https://github.com/anushacodes/tennis-analysis-with-cv) · [tennis tech literatür taraması](https://github.com/hampen2929/survey_on_tennis_tech)
