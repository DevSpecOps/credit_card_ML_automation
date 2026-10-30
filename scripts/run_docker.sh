#!/bin/bash

echo "Building Docker image..."
docker compose build

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Starting container..."
docker compose up -d

echo "Waiting for API to be ready..."
max_retries=10
retry_count=0
ready=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
        ready=1
        break
    else
        sleep 2
        retry_count=$((retry_count + 1))
    fi
done

if [ $ready -eq 1 ]; then
    echo "[OK] API is UP and running at http://127.0.0.1:8000/health"
else
    echo "[FAIL] API failed to start. Check logs with: docker compose logs"
fi
