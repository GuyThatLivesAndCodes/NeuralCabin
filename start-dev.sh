#!/bin/bash
# NeuralCabin — launch the Tauri desktop app in development mode.
#
# This starts both the Vite dev server (hot-reload) AND the Tauri window.
# For a production build run: npm run tauri -- build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧠 NeuralCabin"
echo "=============="
echo ""

for cmd in cargo node npm; do
  if ! command -v "$cmd" &>/dev/null; then
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

if [ ! -d "node_modules" ]; then
  echo "Installing root deps..."
  npm install
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend deps..."
  npm --prefix frontend install
fi

echo "Starting NeuralCabin in dev mode (Tauri + Vite hot-reload)..."
echo "A desktop window will open automatically."
echo "Press Ctrl+C to stop."
echo ""
exec npm run dev
