@echo off
REM ============================================================
REM  Face Recognition System — Windows Setup (Python 3.13)
REM  Run once to create venv313 + install all dependencies.
REM  Requires Python 3.13 (already installed on this machine).
REM ============================================================

echo.
echo [1/5] Creating virtual environment with Python 3.13...
py -3.13 -m venv venv313
if errorlevel 1 (
    echo ERROR: Python 3.13 not found via the py launcher.
    echo Download from: https://www.python.org/downloads/
    pause & exit /b 1
)

echo.
echo [2/5] Activating virtual environment...
call venv313\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Installing TensorFlow (this may take several minutes — ~500 MB)...
pip install "tensorflow>=2.16.0"

echo.
echo [5/5] Installing remaining dependencies...
pip install tf-keras "deepface>=0.0.90" "opencv-python>=4.8.0" "numpy>=1.24.0" "PyYAML>=6.0.0" "Pillow>=10.0.0"

echo.
echo ============================================================
echo  Pre-downloading ArcFace model weights (~500 MB, one-time)...
echo  This may take a few minutes depending on your connection.
echo ============================================================
set PYTHONUTF8=1
python -c "from deepface import DeepFace; import numpy as np; DeepFace.represent(np.zeros((112,112,3),dtype=np.uint8), model_name='ArcFace', enforce_detection=False, detector_backend='skip'); print('ArcFace model ready.')"

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To run the face recognition system:
echo    1. Add face photos to known_faces\whitelist\ or known_faces\blacklist\
echo       (filename = FirstName_LastName.jpg)
echo    2. Open a terminal in this folder and run:
echo         venv313\Scripts\activate
echo         python main.py
echo.
echo  Keyboard shortcuts during capture:
echo    Q / ESC  - Quit
echo    P        - Pause / Resume
echo    R        - Reload face database from disk
echo ============================================================
pause
