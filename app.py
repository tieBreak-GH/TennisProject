import os
import tempfile

import pandas as pd
import streamlit as st

from main import process_video

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
BALL_MODEL = os.path.join(WEIGHTS_DIR, 'ball_track_model.pt')
COURT_MODEL = os.path.join(WEIGHTS_DIR, 'court_model.pt')
BOUNCE_MODEL = os.path.join(WEIGHTS_DIR, 'bounce_model.cbm')

HIGHLIGHT_REASON_LABELS = {'fastest': 'En hızlı vuruş', 'longest': 'En uzun ral(l)i'}

st.set_page_config(page_title='Tenis Video Analizi', page_icon='🎾', layout='centered')
st.title('🎾 Tenis Video Analizi')
st.caption('Topu takip eder, hızını (km/h) hesaplar ve saha/oyuncu bilgisiyle birlikte videonun üzerine çizer.')

missing_models = [p for p in (BALL_MODEL, COURT_MODEL, BOUNCE_MODEL) if not os.path.isfile(p)]
if missing_models:
    st.error('Model ağırlıkları bulunamadı:\n' + '\n'.join(missing_models))
    st.stop()

with st.expander('Video çekim önerileri'):
    st.markdown(
        '- Kamera sabit olmalı (tripod veya sabit bir yere yerleştirin).\n'
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

uploaded_files = st.file_uploader('Video yükleyin (birden fazla seçebilirsiniz)',
                                   type=['mp4', 'mov', 'm4v', 'avi'], accept_multiple_files=True)
detect_persons = st.checkbox('Oyuncuları tespit et', value=True,
                              help='Kapatırsanız en ağır model (oyuncu tespiti) atlanır, işlem belirgin şekilde hızlanır; '
                                   'ama oyuncu kutuları ve minimap noktaları oluşmaz.')
generate_highlights = st.checkbox('Highlight klipleri oluştur', value=False,
                                   help='En hızlı vuruşlu ve en uzun ral(l)ilerin kısa video klipleri otomatik olarak '
                                        'ayrı dosyalar halinde kesilir.')
highlights_top_n = 3
if generate_highlights:
    highlights_top_n = st.slider('Kriter başına klip sayısı', min_value=1, max_value=5, value=3,
                                  help='Örn. 3 seçilirse en fazla 3 "en hızlı vuruşlu" ve 3 "en uzun ral(l)i" klibi '
                                       'kesilir (aynı ral(l)i iki kritere de girebilir).')

if uploaded_files and st.button('Analiz Et', type='primary'):
    overall_status = st.empty()
    progress_bar = st.empty()
    eta_caption = st.empty()

    results = []
    total = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        overall_status.info('Video {}/{}: {}'.format(idx, total, uploaded_file.name))
        work_dir = tempfile.mkdtemp(prefix='tennis_analysis_')
        input_path = os.path.join(work_dir, uploaded_file.name)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        output_path = os.path.join(work_dir, 'output.mp4')

        def on_progress(message, fraction, eta):
            progress_bar.progress(fraction, text=message)
            eta_caption.caption('Tahmini kalan süre: ~' + format_eta(eta) if eta is not None else '')

        try:
            stats = process_video(BALL_MODEL, COURT_MODEL, BOUNCE_MODEL,
                                   input_path, output_path,
                                   detect_persons=detect_persons,
                                   progress_callback=on_progress,
                                   generate_highlights=generate_highlights,
                                   highlights_top_n=highlights_top_n)
            results.append({'name': uploaded_file.name, 'output_path': output_path, 'stats': stats, 'error': None})
        except Exception as e:
            results.append({'name': uploaded_file.name, 'output_path': None, 'stats': None, 'error': str(e)})

    overall_status.empty()
    progress_bar.empty()
    eta_caption.empty()
    st.session_state.results = results

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

            if stats.get('rallies'):
                st.subheader('Ral(l)iler')
                rally_rows = [{
                    'Ral(l)i': rally['rally_no'],
                    'Vuruş sayısı': rally['num_shots'],
                    'Süre (s)': round(rally['duration_s'], 1),
                    'Servis': 'Evet' if any(s['is_serve'] for s in rally['shots']) else '—',
                    'Ort. hız (km/h)': round(rally['avg_speed_kmh']) if rally['avg_speed_kmh'] else None,
                    'Maks. hız (km/h)': round(rally['max_speed_kmh']) if rally['max_speed_kmh'] else None,
                } for rally in stats['rallies']]
                st.dataframe(pd.DataFrame(rally_rows), hide_index=True, use_container_width=True)

            st.video(r['output_path'])

            with open(r['output_path'], 'rb') as f:
                st.download_button('Videoyu indir', f, file_name='tenis_analiz_{}.mp4'.format(idx + 1),
                                    mime='video/mp4', key='dl_video_{}'.format(idx))

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
