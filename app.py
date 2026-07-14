import os
import tempfile

import streamlit as st

from main import process_video

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
BALL_MODEL = os.path.join(WEIGHTS_DIR, 'ball_track_model.pt')
COURT_MODEL = os.path.join(WEIGHTS_DIR, 'court_model.pt')
BOUNCE_MODEL = os.path.join(WEIGHTS_DIR, 'bounce_model.cbm')

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

if 'output_path' not in st.session_state:
    st.session_state.output_path = None
    st.session_state.stats = None

uploaded_file = st.file_uploader('Video yükleyin', type=['mp4', 'mov', 'm4v', 'avi'])
detect_persons = st.checkbox('Oyuncuları tespit et', value=True,
                              help='Kapatırsanız en ağır model (oyuncu tespiti) atlanır, işlem belirgin şekilde hızlanır; '
                                   'ama oyuncu kutuları ve minimap noktaları oluşmaz.')

if uploaded_file is not None and st.button('Analiz Et', type='primary'):
    work_dir = tempfile.mkdtemp(prefix='tennis_analysis_')
    input_path = os.path.join(work_dir, uploaded_file.name)
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.read())
    output_path = os.path.join(work_dir, 'output.mp4')

    status_box = st.empty()

    def on_progress(message):
        status_box.info(message)

    with st.spinner('İşleniyor, bu birkaç dakika sürebilir...'):
        try:
            stats = process_video(BALL_MODEL, COURT_MODEL, BOUNCE_MODEL,
                                   input_path, output_path,
                                   detect_persons=detect_persons,
                                   progress_callback=on_progress)
            st.session_state.output_path = output_path
            st.session_state.stats = stats
            status_box.success('Analiz tamamlandı.')
        except Exception as e:
            status_box.empty()
            st.error(f'İşlem sırasında hata oluştu: {e}')

if st.session_state.output_path and os.path.isfile(st.session_state.output_path):
    stats = st.session_state.stats

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

    st.video(st.session_state.output_path)

    with open(st.session_state.output_path, 'rb') as f:
        st.download_button('Videoyu indir', f, file_name='tenis_analiz.mp4', mime='video/mp4')
