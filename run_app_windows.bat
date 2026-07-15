@echo off
cd /d "%~dp0"

if not exist venv (
    echo Sanal ortam bulunamadi, olusturuluyor...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Bagimliliklar kontrol ediliyor/kuruluyor...
pip install -r requirements.txt

echo Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilirsiniz)...
streamlit run app.py --server.port 8501 --server.headless false

pause
