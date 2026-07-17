import atexit
import glob
import json
import os
import shutil
import tempfile
import threading
import time

import pandas as pd
import streamlit as st

from main import process_video


class AnalysisCancelled(Exception):
    """Raised from the progress callback (see _run_batch) to unwind
    process_video early once the user clicks 'İptal Et' - process_video has
    no cancellation support of its own, but it already calls back into our
    code every ~20 frames, which is a convenient, already-existing hook to
    check a cancellation flag without threading a new parameter through the
    whole pipeline."""

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
BALL_MODEL = os.path.join(WEIGHTS_DIR, 'ball_track_model.pt')
COURT_MODEL = os.path.join(WEIGHTS_DIR, 'court_model.pt')
BOUNCE_MODEL = os.path.join(WEIGHTS_DIR, 'bounce_model.cbm')

HIGHLIGHT_REASON_LABELS = {'fastest': 'En hızlı vuruş', 'longest': 'En uzun ral(l)i'}
LINE_CALL_LABELS = {'in': 'İçeride', 'out': 'Dışarıda', 'belirsiz': 'Belirsiz'}

_TEMP_DIR_PREFIX = 'tennis_analysis_'


def _cleanup_work_dir(work_dir):
    """Remove one analysis's temp dir (input video, output video, highlights, ...)."""
    if work_dir and os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)


def _cleanup_all_temp_dirs():
    """Best-effort sweep of every leftover analysis temp dir, used at process exit."""
    pattern = os.path.join(tempfile.gettempdir(), _TEMP_DIR_PREFIX + '*')
    for path in glob.glob(pattern):
        _cleanup_work_dir(path)


if '_cleanup_registered' not in st.session_state:
    atexit.register(_cleanup_all_temp_dirs)
    st.session_state._cleanup_registered = True


def serve_cell(rally):
    serve_shot = next((s for s in rally['shots'] if s['is_serve']), None)
    if serve_shot is None:
        return '—'
    return LINE_CALL_LABELS.get(serve_shot['line_call'], 'Evet')

st.set_page_config(page_title='Tenis Video Analizi', page_icon='🎾', layout='wide')
st.title('🎾 Tenis Video Analizi')
st.caption('Topu takip eder, hızını (km/h) hesaplar ve saha/oyuncu bilgisiyle birlikte videonun üzerine çizer.')

missing_models = [p for p in (BALL_MODEL, COURT_MODEL, BOUNCE_MODEL) if not os.path.isfile(p)]
if missing_models:
    st.warning('Model ağırlıkları bulunamadı. Uygulamanın çalışabilmesi için model ağırlıklarının indirilmesi gerekmektedir.')
    st.write('Eksik dosyalar:\n' + '\n'.join([f"- {os.path.basename(p)}" for p in missing_models]))
    if st.button('Model Ağırlıklarını İnternetten İndir', type='primary'):
        import requests
        from download_weights import WEIGHTS
        
        def download_with_progress(file_id, destination, filename):
            url = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = session.get(url, params={'id': file_id}, headers=headers, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(url, params=params, headers=headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            written = 0
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            with open(destination, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    written += len(data)
                    if total_size > 0:
                        fraction = min(1.0, written / total_size)
                        progress_bar.progress(fraction)
                        status_text.caption(f"{filename} indiriliyor: %{fraction*100:.1f} ({written / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                    else:
                        status_text.caption(f"{filename} indiriliyor: {written / (1024*1024):.1f} MB")
            progress_bar.empty()
            status_text.empty()

        try:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            for filename, file_id in WEIGHTS.items():
                dest_path = os.path.join(WEIGHTS_DIR, filename)
                if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
                    st.info(f"'{filename}' indiriliyor...")
                    download_with_progress(file_id, dest_path, filename)
            st.success('Tüm model ağırlıkları başarıyla indirildi! Uygulama başlatılıyor...')
            st.rerun()
        except Exception as e:
            st.error(f'İndirme sırasında hata oluştu: {e}')
    st.stop()


with st.expander('Video çekim önerileri'):
    st.markdown(
        '- Kamera sabit olmalı (tripod veya sabit bir yere yerleştirin).\n'
        '- Kamerayı **olabildiğince yüksek ve geride** konumlandırın (ör. baseline arkası, '
        'yükseltilmiş bir yere). Top hızı, topun görüntüdeki konumunun saha zeminine '
        'projeksiyonundan hesaplanır; alçak bir kamera (ör. 1.5-2.5 m tripod) bu yüzden '
        'hızı sistematik olarak **abartır** — kamera ne kadar yüksekteyse ölçüm o kadar '
        'gerçeğe yakın olur.\n'
        '- Yatay (landscape) çekin, saha bütünüyle kadrajda olsun.\n'
        '- Kısa ralli klipleri, uzun maç kayıtlarından çok daha hızlı işlenir.\n'
        '- GPU yoksa (CPU üzerinde) işlem birkaç dakika ile onlarca dakika arası sürebilir.'
    )


def format_eta(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return '{} sn'.format(seconds)
    minutes, secs = divmod(seconds, 60)
    return '{} dk {} sn'.format(minutes, secs)


if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

_DEVICE_LABELS = ['Otomatik', 'CPU (garantili)', 'GPU (CUDA/DirectML/ROCm/MPS dene)']


def _device_for_choice(label):
    """
    Map the radio label to main.process_video's `device` param.
    'Otomatik' and 'GPU dene' both resolve to None (main._select_device
    already tries a GPU backend first and falls back to CPU on its own) -
    the separate 'GPU' option exists so a non-technical user can see and
    pick that intent explicitly, not because it takes a different code path.
    Only 'CPU' forces a specific device, overriding GPU auto-detection.
    """
    return 'cpu' if label.startswith('CPU') else None


def _run_batch(files, device_choice, detect_persons, generate_highlights, highlights_top_n,
                trim_dead_time, cancel_event, progress_state):
    """
    Runs process_video for each file in sequence, on a background thread so
    the main Streamlit script thread stays free to keep rerunning and show
    a live 'İptal Et' button (Streamlit reruns the whole script per
    interaction - a synchronous call here would block that entirely, and a
    click during it would only be handled once process_video returns).

    Must not call any st.* function - there is no Streamlit script-run
    context on a background thread. Progress is instead published by
    mutating progress_state (a plain dict), which the main thread polls and
    renders; cancel_event is checked from on_progress (process_video's own
    progress callback, invoked every ~20 frames) to unwind early via
    AnalysisCancelled - process_video has no cancellation support of its
    own, but this hook already exists and fires often enough to be
    responsive.
    """
    results = []
    total = len(files)
    for idx, (name, data) in enumerate(files, start=1):
        if cancel_event.is_set():
            break
        progress_state['current'] = idx
        progress_state['current_name'] = name
        work_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
        input_path = os.path.join(work_dir, name)
        with open(input_path, 'wb') as f:
            f.write(data)
        output_path = os.path.join(work_dir, 'output.mp4')

        def on_progress(message, fraction, eta):
            if cancel_event.is_set():
                raise AnalysisCancelled()
            progress_state['fraction'] = fraction
            progress_state['message'] = message
            progress_state['eta'] = eta

        try:
            stats = process_video(BALL_MODEL, COURT_MODEL, BOUNCE_MODEL,
                                   input_path, output_path,
                                   device=_device_for_choice(device_choice),
                                   detect_persons=detect_persons,
                                   progress_callback=on_progress,
                                   generate_highlights=generate_highlights,
                                   highlights_top_n=highlights_top_n,
                                   trim_dead_time=trim_dead_time)
            results.append({'name': name, 'work_dir': work_dir, 'output_path': output_path,
                             'stats': stats, 'error': None})
        except AnalysisCancelled:
            _cleanup_work_dir(work_dir)
            progress_state['cancelled'] = True
            break
        except Exception as e:
            results.append({'name': name, 'work_dir': work_dir, 'output_path': None,
                             'stats': None, 'error': str(e)})

    progress_state['results'] = results
    progress_state['done'] = True


with st.form('analiz_formu'):
    uploaded_files = st.file_uploader('Video yükleyin (birden fazla seçebilirsiniz)',
                                       type=['mp4', 'mov', 'm4v', 'avi'], accept_multiple_files=True)
    device_label = st.radio('İşlem birimi', _DEVICE_LABELS, horizontal=True,
                             help='"Otomatik" mevcut en iyi cihazı dener (GPU varsa onu, yoksa CPU). '
                                  '"CPU" her zaman CPU kullanır - GPU sürücüsü/backend deneyseyse (ör. '
                                  'DirectML) sorun çıkarsa buraya alın. "GPU" bir GPU backend\'i zorlar; '
                                  'bulunamazsa yine CPU\'ya düşer. Sonuç panelinde hangi cihazın '
                                  'kullanıldığı gösterilir.')
    detect_persons = st.checkbox('Oyuncuları tespit et', value=True,
                                  help='Kapatırsanız en ağır model (oyuncu tespiti) atlanır, işlem belirgin şekilde hızlanır; '
                                       'ama oyuncu kutuları ve minimap noktaları oluşmaz.')
    generate_highlights = st.checkbox('Highlight klipleri oluştur', value=False,
                                       help='En hızlı vuruşlu ve en uzun ral(l)ilerin kısa video klipleri otomatik olarak '
                                            'ayrı dosyalar halinde kesilir.')
    highlights_top_n = st.slider('Kriter başına klip sayısı', min_value=1, max_value=5, value=3,
                                  help='Sadece "Highlight klipleri oluştur" işaretliyse geçerlidir. Örn. 3 seçilirse '
                                       'en fazla 3 "en hızlı vuruşlu" ve 3 "en uzun ral(l)i" klibi kesilir (aynı '
                                       'ral(l)i iki kritere de girebilir).')
    trim_dead_time = st.checkbox('Ölü zamanı kırp (sadece ral(l)iler)', value=False,
                                  help='Ral(l)i olmayan bölümleri (top değişimi, servis öncesi bekleme vb.) atlayıp '
                                       'sadece oyun anlarını içeren tek bir video oluşturur.')
    submitted = st.form_submit_button('Analiz Et', type='primary', disabled=st.session_state.processing)

if submitted and uploaded_files and not st.session_state.processing:
    # the previous batch's results are about to be replaced once this one
    # finishes, so their temp dirs (input/output video, highlights, ...) are
    # no longer reachable from the UI - safe to free the disk space now
    for prev in st.session_state.results:
        _cleanup_work_dir(prev.get('work_dir'))
    st.session_state.results = []

    files_data = [(f.name, f.getbuffer().tobytes()) for f in uploaded_files]
    st.session_state.cancel_event = threading.Event()
    st.session_state.progress_state = {
        'fraction': 0.0, 'message': '', 'eta': None, 'current': 0, 'current_name': '',
        'total': len(files_data), 'done': False, 'cancelled': False, 'results': None,
    }
    st.session_state.processing = True
    thread = threading.Thread(
        target=_run_batch,
        args=(files_data, device_label, detect_persons, generate_highlights, highlights_top_n,
              trim_dead_time, st.session_state.cancel_event, st.session_state.progress_state),
        daemon=True)
    thread.start()
    st.rerun()

if st.session_state.processing:
    progress_state = st.session_state.progress_state
    st.info('Video {}/{}: {}'.format(progress_state['current'], progress_state['total'],
                                      progress_state['current_name']))
    st.progress(progress_state['fraction'], text=progress_state['message'])
    if progress_state['eta'] is not None:
        st.caption('Tahmini kalan süre: ~' + format_eta(progress_state['eta']))

    if progress_state['done']:
        st.session_state.processing = False
        st.session_state.results = progress_state['results']
        if progress_state['cancelled']:
            st.warning('Analiz iptal edildi. O ana kadar tamamlanan videoların sonuçları aşağıda listelenir.')
        st.rerun()
    else:
        if st.session_state.cancel_event.is_set():
            st.info('İptal ediliyor, geçerli video karesinin işlenmesi bitene kadar sürebilir...')
        elif st.button('İptal Et', type='secondary'):
            st.session_state.cancel_event.set()
        time.sleep(0.5)
        st.rerun()

if st.session_state.results:
    single = len(st.session_state.results) == 1
    for idx, r in enumerate(st.session_state.results):
        with st.expander(r['name'], expanded=single):
            if r['error']:
                st.error('İşlem sırasında hata oluştu: {}'.format(r['error']))
                continue

            stats = r['stats']
            col1, col2, col3 = st.columns(3)
            col1.metric('Kare sayısı', stats['num_frames'])
            col2.metric('Maks. top hızı',
                        '{:.0f} km/h'.format(stats['max_speed_kmh']) if stats['max_speed_kmh'] else '—')
            col3.metric('Sekme sayısı', stats['num_bounces'])
            if stats['avg_speed_kmh']:
                st.caption('Ortalama top hızı: {:.0f} km/h'.format(stats['avg_speed_kmh']))

            device_note = 'Top/saha: {}'.format(stats['ball_court_device'])
            if stats['person_device']:
                device_note += ' · Oyuncu tespiti: {}'.format(stats['person_device'])
            st.caption(device_note)

            # Faz 4: hangi karelerin kamera-yüksekliğinden bağımsız 3B
            # yörünge fitine mi, yoksa eski 2B saha-düzlemi projeksiyonuna mı
            # dayandığını göster - bkz. speed_estimator.estimate_ball_speed.
            num_3d = stats.get('num_frames_3d', 0)
            num_2d = stats.get('num_frames_2d', 0)
            total_method = num_3d + num_2d
            if total_method:
                pct_3d = round(100 * num_3d / total_method)
                if num_3d and num_2d:
                    st.info('Hız yöntemi: karelerin %{}\'i 3B kalibre (kamera ≈ {:.1f} m), '
                            'kalanı 2B tahmin (düşük güven).'.format(
                                pct_3d, stats['camera_height_cm'] / 100))
                elif num_3d:
                    st.success('Hız yöntemi: 3B kalibre — kamera-yüksekliğinden bağımsız '
                                '(kamera ≈ {:.1f} m).'.format(stats['camera_height_cm'] / 100))
                else:
                    st.warning('Hız yöntemi: 2B tahmin (düşük güven) — 3B kalibrasyon bu videoda '
                                'devreye giremedi (saha net görünmüyor, uçuş segmentleri çok kısa, '
                                'veya homografi dejenere). Farklı kamera yüksekliklerinde hız bu '
                                'videoda daha az güvenilir olabilir.')

            if stats.get('rallies'):
                st.subheader('Ral(l)iler')
                rally_rows = [{
                    'Ral(l)i': rally['rally_no'],
                    'Vuruş sayısı': rally['num_shots'],
                    'Süre (s)': round(rally['duration_s'], 1),
                    'Servis': serve_cell(rally),
                    'Ort. hız (km/h)': round(rally['avg_speed_kmh']) if rally['avg_speed_kmh'] else None,
                    'Maks. hız (km/h)': round(rally['max_speed_kmh']) if rally['max_speed_kmh'] else None,
                } for rally in stats['rallies']]
                rally_df = pd.DataFrame(rally_rows)
                st.dataframe(rally_df, hide_index=True, use_container_width=True)

                export_col1, export_col2 = st.columns(2)
                export_col1.download_button(
                    'Ral(l)i verisini CSV indir', rally_df.to_csv(index=False).encode('utf-8'),
                    file_name='rallyler_{}.csv'.format(idx + 1), mime='text/csv',
                    key='dl_csv_{}'.format(idx))
                export_col2.download_button(
                    'Tüm istatistikleri JSON indir', json.dumps(stats, indent=2, ensure_ascii=False).encode('utf-8'),
                    file_name='istatistikler_{}.json'.format(idx + 1), mime='application/json',
                    key='dl_json_{}'.format(idx))

            if stats.get('ball_speed'):
                speed_df = pd.DataFrame({
                    'Zaman (s)': [i / stats['fps'] for i in range(len(stats['ball_speed']))],
                    'Hız (km/h)': stats['ball_speed'],
                }).dropna()
                if not speed_df.empty:
                    st.subheader('Top hızı (zaman içinde)')
                    st.line_chart(speed_df.set_index('Zaman (s)'))
                    if num_3d and num_2d:
                        st.caption('Grafikteki değerlerin %{}\'i 3B kalibre, kalanı 2B tahmin '
                                   'yöntemiyle hesaplandı.'.format(pct_3d))

            st.video(r['output_path'])

            with open(r['output_path'], 'rb') as f:
                st.download_button('Videoyu indir', f, file_name='tenis_analiz_{}.mp4'.format(idx + 1),
                                    mime='video/mp4', key='dl_video_{}'.format(idx))

            if stats.get('rallies_only_video') and os.path.isfile(stats['rallies_only_video']):
                st.subheader('Sadece ral(l)iler (ölü zaman kırpılmış)')
                st.video(stats['rallies_only_video'])
                with open(stats['rallies_only_video'], 'rb') as f:
                    st.download_button('Bu videoyu indir', f, file_name='sadece_rallyler_{}.mp4'.format(idx + 1),
                                        mime='video/mp4', key='dl_rallies_only_{}'.format(idx))

            if stats.get('highlight_clips'):
                st.subheader('Highlight klipleri')
                for clip in stats['highlight_clips']:
                    reason_label = ' + '.join(HIGHLIGHT_REASON_LABELS[reason] for reason in clip['reasons'])
                    st.caption('Ral(l)i {} — {}'.format(clip['rally_no'], reason_label))
                    st.video(clip['path'])
                    with open(clip['path'], 'rb') as f:
                        st.download_button('Highlight indir (ral(l)i {})'.format(clip['rally_no']), f,
                                            file_name=os.path.basename(clip['path']), mime='video/mp4',
                                            key='dl_highlight_{}_{}'.format(idx, clip['rally_no']))
