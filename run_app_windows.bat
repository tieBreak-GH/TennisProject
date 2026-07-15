@echo off
cd /d "%~dp0"

if not exist venv (
    echo Sanal ortam bulunamadi, olusturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo HATA: sanal ortam olusturulamadi. Python kurulu ve PATH'e ekli mi kontrol edin
        echo ^(komut satirinda "python --version" calisiyor mu bakin^).
        pause
        exit /b 1
    )
)

if not exist venv\Scripts\activate.bat (
    echo.
    echo HATA: venv\Scripts\activate.bat bulunamadi, sanal ortam bozuk olabilir.
    echo "venv" klasorunu silip bu betigi tekrar calistirmayi deneyin.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo Bagimliliklar kontrol ediliyor/kuruluyor...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo HATA: bagimlilik kurulumu basarisiz oldu ^(yukaridaki pip hata mesajina bakin^).
    echo Bu genelde internet baglantisi, disk alani veya bir paketin bu Python
    echo surumuyle uyumsuzlugundan kaynaklanir.
    pause
    exit /b 1
)

echo Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilirsiniz)...
streamlit run app.py --server.port 8501 --server.headless false

pause
