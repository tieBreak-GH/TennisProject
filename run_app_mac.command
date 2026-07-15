#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Sanal ortam bulunamadi, olusturuluyor..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Bagimliliklar kontrol ediliyor/kuruluyor..."
pip install -r requirements.txt

echo "Model ağırlıkları kontrol ediliyor ve eksikse indiriliyor..."
python download_weights.py || {
    echo ""
    echo "UYARI: Model ağırlıkları otomatik indirilemedi. İnternet bağlantınızı kontrol edin."
    read -p "Devam etmek için Enter tuşuna basın..."
}

echo "Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilir ya da Ctrl+C basabilirsiniz)..."
streamlit run app.py --server.port 8501 --server.headless false

