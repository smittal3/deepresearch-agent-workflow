@echo off
REM Launch the app on Windows. Run:  run.bat
if not exist ".venv\Scripts\activate.bat" (
  echo No virtual environment found. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
