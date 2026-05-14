#!/bin/bash
# NeuralCabin Development Startup Script
# Starts both the Rust backend and React frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧠 NeuralCabin Development Environment"
echo "======================================"
echo ""

# Check if required tools are installed
if ! command -v cargo &> /dev/null; then
  echo "❌ Rust/Cargo not found. Please install from https://rustup.rs/"
  exit 1
fi

if ! command -v node &> /dev/null; then
  echo "❌ Node.js not found. Please install from https://nodejs.org/"
  exit 1
fi

if ! command -v npm &> /dev/null; then
  echo "❌ npm not found. Please install Node.js from https://nodejs.org/"
  exit 1
fi

echo "✅ Rust/Cargo found"
echo "✅ Node.js/npm found"
echo ""

# Backend setup
echo "Setting up backend..."
if [ ! -f "target/debug/neuralcabin-backend" ]; then
  echo "Building backend (first time, this may take a minute)..."
  cargo build --package neuralcabin-backend
fi
echo "✅ Backend ready"
echo ""

# Frontend setup
echo "Setting up frontend..."
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  cd frontend
  npm install
  cd ..
fi
echo "✅ Frontend ready"
echo ""

# Start backend
echo "🚀 Starting backend on http://127.0.0.1:3001..."
cargo run --package neuralcabin-backend &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
for i in {1..30}; do
  if curl -s http://localhost:3001/api/networks &> /dev/null; then
    echo "✅ Backend is ready!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID
    exit 1
  fi
  sleep 1
done

echo ""

# Start frontend
echo "🎨 Starting frontend on http://localhost:5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "======================================"
echo "✅ NeuralCabin is running!"
echo "======================================"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔌 Backend:  http://127.0.0.1:3001"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Cleanup function
cleanup() {
  echo ""
  echo "Stopping services..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  echo "Goodbye! 👋"
}

trap cleanup EXIT

# Wait for both processes
wait
