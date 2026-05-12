@echo off
REM Quick start script WITHOUT Redis requirement
echo ========================================
echo Stock Manipulation Detection System
echo ========================================
echo.

REM Check if MongoDB is running
echo [1/2] Checking MongoDB...
tasklist /FI "IMAGENAME eq mongod.exe" 2>NUL | find /I /N "mongod.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo   ✓ MongoDB is running
) else (
    echo   ✗ MongoDB is NOT running
    echo   Please start MongoDB before continuing
    echo.
    echo   Option 1: If MongoDB is installed as a service:
    echo     net start MongoDB
    echo.
    echo   Option 2: If MongoDB is not a service:
    echo     mongod
    echo.
    pause
    exit /b 1
)

echo [2/2] Starting Flask API...
echo.
echo NOTE: Redis is optional. System will use in-memory cache if Redis is not available.
echo.
echo ========================================
echo API will start on http://localhost:5000
echo Dashboard: Open frontend\index.html
echo ========================================
echo.

cd api
python app.py
