@echo off
title KnowledgeForge
cd /d "%~dp0"

echo ============================================
echo   Starting KnowledgeForge
echo ============================================
echo.

echo [1/2] Starting Ollama (AI engine)...
start "Ollama" /min ollama serve
timeout /t 3 /nobreak >nul

echo [2/2] Starting the app...
call venv\Scripts\activate.bat
echo.
echo App will open at http://localhost:8501
echo Keep this window open. Press Ctrl+C to stop.
echo.
streamlit run app.py
