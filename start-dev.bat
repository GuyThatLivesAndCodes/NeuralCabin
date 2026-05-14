@echo off
REM NeuralCabin Development Startup Script for Windows
REM Starts both the Rust backend and React frontend

setlocal enabledelayedexpansion

echo 🧠 NeuralCabin Development Environment
echo ======================================
echo.

REM Check if cargo is installed
where cargo >nul 2>nul
if errorlevel 1 (
  echo ❌ Rust/Cargo not found. Please install from https://rustup.rs/
  pause
  exit /b 1
)

REM Check if node is installed
where node >nul 2>nul
if errorlevel 1 (
  echo ❌ Node.js not found. Please install from https://nodejs.org/
  pause
  exit /b 1
)

REM Check if npm is installed
where npm >nul 2>nul
if errorlevel 1 (
  echo ❌ npm not found. Please install Node.js from https://nodejs.org/
  pause
  exit /b 1
)

echo ✅ Rust/Cargo found
echo ✅ Node.js/npm found
echo.

REM Backend setup
echo Setting up backend...
if not exist "target\debug\neuralcabin-backend.exe" (
  echo Building backend (first time, this may take a minute)...
  call cargo build --package neuralcabin-backend
)
echo ✅ Backend ready
echo.

REM Frontend setup
echo Setting up frontend...
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd frontend
  call npm install
  cd ..
)
echo ✅ Frontend ready
echo.

REM Start backend in new window
echo 🚀 Starting backend on http://127.0.0.1:3001...
start "NeuralCabin Backend" cargo run --package neuralcabin-backend

REM Wait a bit for backend to start
timeout /t 3 /nobreak

REM Start frontend in new window
echo 🎨 Starting frontend on http://localhost:5173...
cd frontend
start "NeuralCabin Frontend" npm run dev
cd ..

echo.
echo ======================================
echo ✅ NeuralCabin is running!
echo ======================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔌 Backend:  http://127.0.0.1:3001
echo.
echo Close both windows to stop the services.
echo.
pause
