@echo off
REM One-time setup for Windows. Double-click or run:  setup.bat
REM Requires Python 3.10+ (3.11 recommended) from python.org with "Add to PATH" checked.

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv || python -m venv .venv
) else (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete.
echo Next: copy .env.example to .env and add your keys (or paste them in the app sidebar):
echo   - OPENROUTER_API_KEY  (required)  https://openrouter.ai/keys
echo   - TAVILY_API_KEY      (optional, for web search)  https://app.tavily.com
echo Then run:  run.bat
pause
