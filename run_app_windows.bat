@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist venv (
    rem scipy/torch only ship Windows wheels up to Python 3.12 (as of this
    rem writing) - a newer default "python" (e.g. 3.13/3.14) forces pip to
    rem compile from source, which fails without a Fortran compiler. Prefer
    rem the py launcher to pick a known-compatible version if one is
    rem installed, newest first, before falling back to plain "python".
    set PYTHON_CMD=
    for %%V in (3.12 3.11 3.10 3.9) do (
        if not defined PYTHON_CMD (
            py -%%V -c "" >nul 2>&1
            if not errorlevel 1 set PYTHON_CMD=py -%%V
        )
    )
    if not defined PYTHON_CMD (
        python -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
        if not errorlevel 1 set PYTHON_CMD=python
    )
    if not defined PYTHON_CMD (
        echo.
        echo HATA: uygun bir Python bulunamadi. Bu proje Python 3.9-3.12
        echo arasini destekliyor ^(3.13/3.14 icin scipy/torch wheel'leri henuz yok^).
        echo https://www.python.org/downloads/ adresinden Python 3.12 kurun.
        pause
        exit /b 1
    )

    echo Sanal ortam bulunamadi, !PYTHON_CMD! ile olusturuluyor...
    !PYTHON_CMD! -m venv venv
    if errorlevel 1 (
        echo.
        echo HATA: sanal ortam olusturulamadi.
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

echo Model agirliklari kontrol ediliyor ve eksikse indiriliyor...
python download_weights.py
if errorlevel 1 (
    echo.
    echo UYARI: Model agirliklari indirilirken bir sorun olustu.
    echo Internet baglantinizi kontrol edin veya agirliklari manuel indirin.
    pause
)

echo Web arayuzu baslatiliyor (kapatmak icin bu pencereyi kapatabilirsiniz)...
streamlit run app.py --server.port 8501 --server.headless false

pause
