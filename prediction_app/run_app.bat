@echo off
echo Starting Medical Claim Cost Predictor...
cd /d "%~dp0"
streamlit run app.py
pause
