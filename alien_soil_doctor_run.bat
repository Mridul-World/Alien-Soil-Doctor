@echo off
setlocal

:: === CONFIGURATION ===
set "ENV_NAME=tf210-gpu"
set "CONDA_PATH=C:\Users\shubh\anaconda3"
set "PROJECT_DIR=C:\Work_place\isru project v1"
set "LOG_DIR=%PROJECT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\runlog.txt"
set "UI_DIR=%PROJECT_DIR%\ui of model"

echo 🧠 Launching Alien Soil Doctor...
echo ==============================

:: ✅ Step 1: Activate conda environment
echo 🔄 Activating environment: %ENV_NAME%
call "%CONDA_PATH%\Scripts\activate" %ENV_NAME%

:: ✅ Step 2: Navigate to project directory
echo 📂 Navigating to: %PROJECT_DIR%
cd /d "%PROJECT_DIR%"

:: ✅ Step 3: Ensure logs directory exists
if not exist "%LOG_DIR%" (
    echo 📁 Creating logs directory...
    mkdir "%LOG_DIR%"
)

:: ✅ Step 4: Run model evaluation
echo 🚀 Running evaluate_model.py...
python evaluate_model.py > "%LOG_FILE%" 2>&1

:: ✅ Step 5: Launch Streamlit UI in browser
echo 🌐 Launching Streamlit UI...
cd /d "%UI_DIR%"
start http://localhost:8501
streamlit run app.py >> "%LOG_FILE%" 2>&1

echo ✅ All tasks completed! Logs saved to: %LOG_FILE%
pause
