@echo off
REM Start background services
echo ========================================
echo Starting Background Services
echo ========================================
echo.
echo This will start:
echo - Live market data streaming (Alpaca)
echo - Social media monitoring
echo - Automatic predictions and alerts
echo.
echo Press Ctrl+C to stop
echo.

python scripts\main_service.py
