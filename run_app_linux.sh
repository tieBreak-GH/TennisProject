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

echo "Web arayuzu baslatiliyor (kapatmak icin Ctrl+C basabilirsiniz)..."
(sleep 3 && (command -v xdg-open >/dev/null 2>&1 && xdg-open "http://localhost:8501" || echo "Tarayicinizda acin: http://localhost:8501")) &
streamlit run app.py --server.headless true --server.port 8501
