@echo off
setlocal
cd /d "%~dp0"
python -c "import cv2, tkinter" >nul 2>&1
if errorlevel 1 (
    echo Python 3 with OpenCV is required.
    echo Install Python, then run: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
python app.py
if errorlevel 1 pause
