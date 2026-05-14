@echo off
REM NeuralCabin — launch the Tauri desktop app in development mode.
REM This starts both the Vite dev server (hot-reload) AND the Tauri window.
REM For a production build run: npm run tauri -- build

setlocal

echo NeuralCabin
echo ===========
echo.

where cargo >nul 2>nul || (echo Rust/Cargo not found. Install from https://rustup.rs/ && pause && exit /b 1)
where npm   >nul 2>nul || (echo Node.js/npm not found. Install from https://nodejs.org/ && pause && exit /b 1)

if not exist "node_modules" (
  echo Installing root dependencies...
  call npm install
)

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  npm --prefix frontend install
)

echo Starting NeuralCabin in dev mode (Tauri + Vite hot-reload)...
echo A desktop window will open automatically.
echo Press Ctrl+C to stop.
echo.
npm run dev
