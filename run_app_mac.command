#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Sanal ortam bulunamadi, olusturuluyor..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilir ya da Ctrl+C basabilirsiniz)..."
(sleep 3 && open "http://localhost:8501") &
streamlit run app.py --server.headless true --server.port 8501
