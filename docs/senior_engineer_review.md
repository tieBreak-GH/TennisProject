# TennisProject — Kıdemli Yazılım Mühendisi Teknik, Mantık ve UX/UI Değerlendirme Raporu

Bu rapor, `TennisProject` (Tenis Analiz Sistemi) kod tabanını kıdemli bir yazılım mühendisi gözüyle teknik mimari, algoritmik mantık ve kullanıcı deneyimi (UX/UI) boyutlarında incelemektedir. Mevcut sistemin güçlü yönlerini, kritik darboğazlarını ve geliştirme önerilerini içermektedir.

---

## 1. Teknik ve Mimari Değerlendirme

### 1.1. Streaming Pipeline ve Video İşleme Verimliliği
* **Durum:** `main.py` içindeki iki geçişli (two-pass) streaming tasarımı (`analyze_streaming` ve `render_streaming`) bellek tüketimini sınırlandırmak açısından çok başarılıdır. OOM (Out Of Memory) hatalarını engeller.
* **Darboğaz (Çoklu Decode):** Video dosyası şu an uçtan uca işlemede **3 kez** decode edilmektedir:
  1. `scene_detect` (PySceneDetect kütüphanesi sahne geçişlerini bulmak için videonun tamamını okur).
  2. `analyze_streaming` (Model çıkarımları ve ham takip verisi üretimi için).
  3. `render_streaming` (Üretilen analiz verisini videonun üzerine çizip diske yazmak için).
  * *Senior Görüşü:* Kısa videolarda PySceneDetect'in decode süresi önemsiz olsa da (ör. 30 saniyede ~0.24 sn), 1-2 saatlik tam maç kayıtlarında bu 3 kat decode süresi işlem süresini ciddi oranda uzatacaktır.
* **Çözüm Önerisi:** PySceneDetect yerine, Pass 1 (`analyze_streaming`) sırasında çerçevelerin renk histogramlarındaki değişimleri OpenCV yardımıyla kendimiz hesaplayarak sahne geçişlerini dinamik olarak tespit edebiliriz. Böylece video 3 kez yerine 2 kez decode edilir ve işlem süresi ~%30 kısalır.

### 1.2. Concurrency (Eşzamanlılık) ve I/O Darboğazları
* **Durum:** Tüm süreç tek bir iş parçacığında (single-threaded) sıralı çalışmaktadır. Çerçeve okunur -> Model çalıştırılır -> Sonraki çerçeveye geçilir.
* **Darboğaz:** GPU/CPU model çalıştırırken OpenCV video okuyucu ve disk yazıcı (VideoWriter) boşta beklemektedir. Benzer şekilde, video I/O işlemleri yapılırken de işlemci (inference) atıl kalmaktadır.
* **Çözüm Önerisi:** Bir **Producer-Consumer** mimarisi kurulabilir.
  * **Thread 1 (Reader):** Çerçeveleri okur ve bir kuyruğa (Queue) atar.
  * **Thread 2 (Inference/Process):** Kuyruktan çerçeveleri alır, modelleri çalıştırır.
  * **Thread 3 (Writer):** İşlenen ve üzerine çizim yapılan çerçeveleri diske yazar.
  Bu paralel yapı, CPU/GPU ve Disk I/O arasındaki uyuşmazlığı gidererek işlem hızını ~%20-40 oranında artırabilir.

### 1.3. Bellek Sızıntısı Riski (Temp Directory Cleanup)
* **Durum:** `app.py` üzerinde her analiz için `tempfile.mkdtemp` ile geçici bir dizin açılmakta ve analiz edilen video buraya kaydedilmektedir.
* **Hata:** Analiz bittikten sonra veya Streamlit oturumu sonlandığında bu geçici dizinler **hiçbir şekilde temizlenmemektedir**.
* **Risk:** Kullanıcılar web arayüzünden 1 GB sınırında birden fazla büyük video yüklediğinde, sunucunun diski çok kısa sürede dolabilir ve sistem çökmelerine yol açabilir.
* **Çözüm Önerisi:** 
  * Streamlit oturumu kapandığında (`session_state` temizlenirken) veya yeni bir analiz başlatılmadan önce eski temp klasörlerini temizleyen bir rutin eklenmelidir.
  * Python `atexit` modülü ile uygulama kapanırken temizlik garanti altına alınmalıdır.

---

## 2. Mantıksal ve Algoritmik Değerlendirme

### 2.1. Top Hızı Hesaplama ve Yörünge Yumuşatma (`speed_estimator.py`)
* **Mantık Hatası (Bounce Kesişimleri):** Topun hızı, çerçeveler arası Öklid mesafesinin homografiyle gerçek dünyaya yansıtılmasıyla hesaplanır. Gürültüyü azaltmak için `BALL_SPEED_WINDOW_FRAMES` (varsayılan 5 çerçeve) boyutunda bir pencere kullanılır.
  * Ancak, topun yön değiştirdiği **sekme (bounce) veya raketle vurulma anlarında**, bu 5 çerçevelik pencere sekme anını ortalayarak iki yönlü hareketi doğrusal bir çizgi gibi birleştirir (V-şeklini düzleştirir).
  * Bu durum, **en yüksek hızın oluştuğu vuruş/sekme anlarında top hızının gerçekte olduğundan daha düşük ölçülmesine** neden olur.
* **Çözüm Önerisi:** Hız hesabı yapılırken sekme (bounce) çerçeveleri birer sınır olarak kabul edilmeli ve hız penceresi sekme noktalarını aşmamalıdır (yani hız hesabı her sekme segmenti içinde sınırlandırılmalıdır).

### 2.2. Servis Kararı ve Çizgi Kontrolü (`rally_analyzer.py`)
* **Sınırlandırma (Doubles/Çiftler Maçı Uyumsuzluğu):** `filter_players` fonksiyonu, sahadaki oyuncuları süzmek için top/orta servis çizgisine (`middle_line`) en yakın tek bir üst ve tek bir alt oyuncuyu seçer.
  * Bu mantık tekler (singles) maçı için mükemmel çalışsa da, **çiftler (doubles) maçlarında sahadaki diğer iki oyuncuyu sessizce eler** ve analizi bozar.
* **Çizgi Kararı Sınırları:** Servis çizgi kararı (`line_call`) sadece servis vuruşlarıyla sınırlandırılmıştır ve bu doğru bir yaklaşımdır (genel ralli çizgi kararları için homografi hassasiyeti yetersizdir).
  * Ancak, homografi matrisinin kamera açısına ve kalibrasyon kalitesine bağlı olarak hata payı (`LINE_CALL_MARGIN_CM = 20`) çok kritiktir. 20 cm tenis için çok geniş bir "belirsiz" (uncertain) alan demektir. Bu durum kullanıcıya net şekilde açıklanmalıdır.

---

## 3. UX/UI ve Kullanıcı Deneyimi Değerlendirmesi

### 3.1. Streamlit Form Tasarımı ve Kazara Yeniden Çalışma (Rerun) Riski
* **Durum:** `app.py` üzerindeki tüm parametre ayarları (`detect_persons`, `generate_highlights`, vb.) ve dosya yükleyici doğrudan arayüze yerleştirilmiştir.
* **Hata:** Streamlit çalışma yapısı gereği, kullanıcı herhangi bir checkbox'a tıkladığında veya slider'ı kaydırdığında tüm betik (script) yukarıdan aşağıya yeniden çalışır. 
  * Eğer bir analiz devam ederken kullanıcı yanlışlıkla bu ayarlardan birini değiştirirse, Streamlit WebSocket bağlantısı üzerinden betiği kesecek ve **çalışmakta olan uzun analiz yarıda kalıp kaybolacaktır**.
* **Çözüm Önerisi:** Tüm ayarlar ve yükleme bileşeni bir `st.form("analiz_formu")` içine alınmalıdır. Kullanıcı ayarları yapıp sadece form içindeki "Analiz Et" (`st.form_submit_button`) butonuna bastığında işlem tetiklenmelidir. Bu sayede işlem esnasında diğer widget'larla etkileşime girilse bile çalışma kesilmeyecektir.

### 3.2. Geniş Ekran Desteği (Layout Optimizasyonu)
* **Durum:** Sayfa düzeni `layout='centered'` olarak ayarlanmıştır.
* **Eleştiri:** Sayfa ortalandığında, özellikle yan yana sütunlar (`st.columns`), ralli tabloları (`st.dataframe`) ve hız grafikleri çok dar bir alana sıkışmaktadır. Kullanıcı analiz videosunu izlerken aynı anda grafiği ve tabloyu okumakta zorlanacaktır.
* **Çözüm Önerisi:** `st.set_page_config(layout='wide')` seçeneğine geçilerek ekranın tamamı kullanılmalıdır. Sol tarafa analiz parametreleri ve temel metrikler, sağ tarafa ise video önizleme, grafikler ve tablolar yerleştirilerek daha dengeli ve profesyonel bir gösterge paneli (dashboard) oluşturulabilir.

### 3.3. Mobil Kullanılabilirlik ve WebSocket Kararlılığı
* **Durum:** Mobil tarayıcılarda arayüz responsive'dir. `avc1` (H.264) codec güncellemesi sayesinde mobil tarayıcılarda video önizleme sorunsuz çalışmaktadır.
* **Sorun (Oturum Kopması):** Mobil cihazlarda tarayıcı arka plana alındığında veya ekran kilitlendiğinde WebSocket bağlantısı hemen kopar. Streamlit bu durumda oturumu sıfırlar ve kullanıcının dakikalarca beklediği analiz sonucu kaybolur.
* **Çözüm Önerisi:** Sonuçlar geçici bir veritabanında veya diskte video ID'si ile eşleştirilerek saklanmalıdır. Oturum kopsa bile kullanıcı sayfayı yenilediğinde "Devam Eden Analizler" veya "Sonuçları Göster" paneli üzerinden kaldığı yerden sonuçlara erişebilmelidir.

---

## 4. Önceliklendirilmiş Geliştirme Yol Haritası

Aşağıdaki tablo, projenin senior seviyesinde bir ürüne dönüşmesi için yapılması gereken geliştirmeleri öncelik sırasına göre listelemektedir:

| Sıra | İyileştirme | Kategori | Zorluk | Etki | Açıklama |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Temp Dizin Temizliği** | Teknik | Kolay | Kritik | Disk dolmasını engellemek için atexit ve oturum bazlı otomatik temizlik rutini. |
| **2** | **Form Tasarımına Geçiş** | UX/UI | Kolay | Yüksek | Analiz sırasında kazara widget tıklamalarıyla işlemin kesilmesini önlemek. |
| **3** | **Layout Genişletme (Wide)** | UX/UI | Kolay | Orta | Dashboard bileşenlerini daha ferah ve okunabilir kılmak. |
| **4** | **Bounce Sınırında Hız Kontrolü** | Mantık | Orta | Yüksek | Sekme/vuruş anlarında top hızının yapay olarak düşmesini önlemek. |
| **5** | **Producer-Consumer Pipeline** | Teknik | Zor | Yüksek | Video okuma, model çalıştırma ve yazma işlerini paralelleştirerek ~%30 hızlanma. |
| **6** | **Sahne Algılamayı Pass 1'e Gömme** | Teknik | Zor | Orta | PySceneDetect'in bağımsız video decode geçişini kaldırıp 3 decode'u 2'ye düşürmek. |
| **7** | **Çiftler (Doubles) Desteği** | Mantık | Orta | Düşük | `filter_players` mantığını genişleterek korttaki 4 oyuncuyu da takip edebilmek. |

---

> [!TIP]
> **Öncelikli Öneri:** Projenin ticari bir servis olarak dağıtılması veya birden fazla kullanıcıya açılması planlanıyorsa, **1. ve 2. maddeler (Temp temizliği ve Form kontrolü)** hemen uygulanmalıdır. Bu iki küçük dokunuş, sistem kararlılığını ve kullanıcı deneyimini doğrudan etkileyen en kritik zayıf noktalardır.
