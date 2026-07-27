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

:: Skip Streamlit's first-run "Welcome / Email" prompt. Without this it waits
:: for keyboard input and never starts the server, so the browser can't connect.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo App will open at http://localhost:8501
echo Keep this window open. Press Ctrl+C to stop.
echo.
streamlit run app.py --server.headless=false
