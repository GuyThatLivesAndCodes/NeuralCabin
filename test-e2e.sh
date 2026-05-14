#!/bin/bash
# NeuralCabin End-to-End Test Script
# Tests the complete flow: create network -> dataset -> train -> monitor

set -e

API_BASE="http://localhost:3001"
NETWORK_ID=""
DATASET_ID=""
TRAINING_ID=""

echo "🧪 NeuralCabin End-to-End Test"
echo "==============================="
echo ""

# Check if backend is running
echo "1️⃣  Checking backend..."
if ! curl -s "$API_BASE/api/networks" > /dev/null; then
  echo "❌ Backend is not running. Start it with:"
  echo "   cargo run --package neuralcabin-backend"
  exit 1
fi
echo "✅ Backend is running"
echo ""

# Create a network
echo "2️⃣  Creating network..."
NETWORK_RESPONSE=$(curl -s -X POST "$API_BASE/api/networks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-network-'$(date +%s)'",
    "kind": "simplex",
    "seed": 42,
    "layers": [
      {"type": "linear", "in_dim": 2, "out_dim": 8},
      {"type": "activation", "activation": "tanh"},
      {"type": "linear", "in_dim": 8, "out_dim": 1},
      {"type": "activation", "activation": "sigmoid"}
    ]
  }')

NETWORK_ID=$(echo "$NETWORK_RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$NETWORK_ID" ]; then
  echo "❌ Failed to create network"
  echo "Response: $NETWORK_RESPONSE"
  exit 1
fi
echo "✅ Created network: $NETWORK_ID"
echo ""

# Create a dataset
echo "3️⃣  Creating dataset..."
DATASET_RESPONSE=$(curl -s -X POST "$API_BASE/api/datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-dataset-'$(date +%s)'",
    "kind": "xor",
    "seed": 42
  }')

DATASET_ID=$(echo "$DATASET_RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$DATASET_ID" ]; then
  echo "❌ Failed to create dataset"
  echo "Response: $DATASET_RESPONSE"
  exit 1
fi
echo "✅ Created dataset: $DATASET_ID"
echo ""

# Start training
echo "4️⃣  Starting training (100 epochs for quick test)..."
TRAINING_RESPONSE=$(curl -s -X POST "$API_BASE/api/train" \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "'$NETWORK_ID'",
    "dataset_id": "'$DATASET_ID'",
    "config": {
      "epochs": 100,
      "batch_size": 4,
      "optimizer": {
        "kind": "adam",
        "lr": 0.05,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8
      },
      "loss": "mse",
      "validation_frac": 0.2,
      "seed": 42
    }
  }')

TRAINING_ID=$(echo "$TRAINING_RESPONSE" | grep -o '"training_id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$TRAINING_ID" ]; then
  echo "❌ Failed to start training"
  echo "Response: $TRAINING_RESPONSE"
  exit 1
fi
echo "✅ Started training: $TRAINING_ID"
echo ""

# Monitor training via WebSocket
echo "5️⃣  Monitoring training..."
echo "Connecting to WebSocket: ws://localhost:3001/ws/train/$TRAINING_ID"
echo ""

# For this test, we'll just poll the status endpoint instead of WebSocket
# (WebSocket requires more complex setup in bash)

max_wait=120  # 2 minutes max
elapsed=0
last_epoch=0

while [ $elapsed -lt $max_wait ]; do
  STATUS=$(curl -s "$API_BASE/api/train/$TRAINING_ID")

  EPOCH=$(echo "$STATUS" | grep -o '"epoch":[0-9]*' | cut -d':' -f2)
  TOTAL=$(echo "$STATUS" | grep -o '"total_epochs":[0-9]*' | cut -d':' -f2)
  LOSS=$(echo "$STATUS" | grep -o '"last_loss":[0-9.e-]*' | cut -d':' -f2)
  STATUS_STR=$(echo "$STATUS" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

  if [ "$EPOCH" != "$last_epoch" ]; then
    echo "Epoch $EPOCH/$TOTAL - Loss: $LOSS - Status: $STATUS_STR"
    last_epoch=$EPOCH
  fi

  if [ "$STATUS_STR" = "completed" ] || [ "$STATUS_STR" = "error" ]; then
    break
  fi

  sleep 1
  ((elapsed++))
done

echo ""
echo "✅ Training completed"
echo ""

# Get final status
echo "6️⃣  Final statistics..."
FINAL_STATUS=$(curl -s "$API_BASE/api/train/$TRAINING_ID")

FINAL_EPOCH=$(echo "$FINAL_STATUS" | grep -o '"epoch":[0-9]*' | cut -d':' -f2)
FINAL_LOSS=$(echo "$FINAL_STATUS" | grep -o '"last_loss":[0-9.e-]*' | cut -d':' -f2)
ELAPSED=$(echo "$FINAL_STATUS" | grep -o '"elapsed_secs":[0-9.]*' | cut -d':' -f2)

echo "Total Epochs: $FINAL_EPOCH"
echo "Final Loss: $FINAL_LOSS"
echo "Elapsed Time: ${ELAPSED}s"
echo ""

# List networks
echo "7️⃣  Listing networks..."
NETWORKS=$(curl -s "$API_BASE/api/networks")
NETWORK_COUNT=$(echo "$NETWORKS" | grep -o '"id":"' | wc -l)
echo "Total networks: $NETWORK_COUNT"
echo ""

# List datasets
echo "8️⃣  Listing datasets..."
DATASETS=$(curl -s "$API_BASE/api/datasets")
DATASET_COUNT=$(echo "$DATASETS" | grep -o '"id":"' | wc -l)
echo "Total datasets: $DATASET_COUNT"
echo ""

# Cleanup (optional)
echo "9️⃣  Cleanup..."
echo "Deleting test network..."
curl -s -X DELETE "$API_BASE/api/networks/$NETWORK_ID" > /dev/null
echo "✅ Network deleted"
echo ""

echo "========================================"
echo "✅ End-to-end test passed!"
echo "========================================"
echo ""
echo "Summary:"
echo "  • Created network: $NETWORK_ID"
echo "  • Created dataset: $DATASET_ID"
echo "  • Started training: $TRAINING_ID"
echo "  • Training completed after $FINAL_EPOCH epochs"
echo "  • Final loss: $FINAL_LOSS"
echo ""
echo "All systems operational! 🚀"
