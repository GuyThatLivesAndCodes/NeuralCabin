@echo off
REM NeuralCabin Startup Script for Windows
REM Builds the frontend then starts the single combined server.

setlocal

echo NeuralCabin
echo ===========
echo.

where cargo >nul 2>nul || (echo Rust/Cargo not found. Install from https://rustup.rs/ && pause && exit /b 1)
where npm   >nul 2>nul || (echo Node.js/npm not found. Install from https://nodejs.org/ && pause && exit /b 1)

REM Install frontend deps if needed
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd frontend && call npm install && cd ..
)

REM Build frontend so assets are current
echo Building frontend...
cd frontend && call npm run build && cd ..
echo Frontend ready.
echo.

REM Build + run the combined server
echo Starting NeuralCabin on http://localhost:3001 ...
echo Your browser will open automatically.
echo Press Ctrl+C to stop.
echo.
cargo run --package neuralcabin-backend --release
