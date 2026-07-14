@echo off
cd /d "%~dp0"

if not exist venv (
    echo Sanal ortam bulunamadi, olusturuluyor...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilirsiniz)...
start "" /min cmd /c "timeout /t 3 >nul & start http://localhost:8501"
streamlit run app.py --server.headless true --server.port 8501

pause
