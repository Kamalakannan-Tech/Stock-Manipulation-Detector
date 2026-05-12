@echo off
REM StockGuard AI — Master Run Script
REM Runs the full data collection → preprocessing → training → evaluation → API pipeline

echo.
echo  ██████████████████████████████████████
echo   StockGuard AI — Full Pipeline Runner
echo  ██████████████████████████████████████
echo.

set PYTHON=venv\Scripts\python.exe

REM ── Step 1: Collect market data ──────────────────────────────────────────────
echo [1/5] Collecting market data (3 months, 7 tickers)...
%PYTHON% scripts/02_collect_historical_data.py --tickers GME AMC TSLA AAPL MSFT NVDA SPY --months 3
if %ERRORLEVEL% NEQ 0 (echo ERROR: Data collection failed & pause & exit /b 1)
echo     Done.

REM ── Step 2: Preprocess ────────────────────────────────────────────────────────
echo.
echo [2/5] Preprocessing features and labeling...
%PYTHON% scripts/03_preprocess_data.py
if %ERRORLEVEL% NEQ 0 (echo ERROR: Preprocessing failed & pause & exit /b 1)
echo     Done.

REM ── Step 3: Train model ───────────────────────────────────────────────────────
echo.
echo [3/5] Training Temporal Fusion Transformer (30 epochs)...
%PYTHON% scripts/04_train_model.py --epochs 30 --batch-size 32 --lr 0.001
if %ERRORLEVEL% NEQ 0 (echo ERROR: Training failed & pause & exit /b 1)
echo     Done.

REM ── Step 4: Evaluate ─────────────────────────────────────────────────────────
echo.
echo [4/5] Evaluating model on test set...
%PYTHON% scripts/05_evaluate_model.py
if %ERRORLEVEL% NEQ 0 (echo WARNING: Evaluation had errors, continuing...)
echo     Done.

REM ── Step 5: Start API ─────────────────────────────────────────────────────────
echo.
echo [5/5] Starting Flask API on http://localhost:5000 ...
echo       Dashboard: open frontend/index.html in your browser
echo.
%PYTHON% api/app.py
