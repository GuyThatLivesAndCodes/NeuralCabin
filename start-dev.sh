#!/bin/bash
# NeuralCabin Development Startup Script
#
# Production use:  just run the binary — the frontend is embedded inside.
# Dev use:         this script rebuilds the frontend then launches the server.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧠 NeuralCabin"
echo "=============="
echo ""

# Check prerequisites
for cmd in cargo node npm; do
  if ! command -v "$cmd" &> /dev/null; then
    case "$cmd" in
      cargo) url="https://rustup.rs/" ;;
      *)     url="https://nodejs.org/" ;;
    esac
    echo "❌ $cmd not found. Install from $url"
    exit 1
  fi
done
echo "✅ Rust + Node.js ready"
echo ""

# Install frontend deps if needed
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

# Rebuild the frontend (so the embedded assets are current)
echo "Building frontend..."
(cd frontend && npm run build)
echo "✅ Frontend built"
echo ""

# Build & run the backend (which now serves the frontend too)
echo "Starting NeuralCabin on http://localhost:3001 ..."
echo "Press Ctrl+C to stop."
echo ""
exec cargo run --package neuralcabin-backend --release
