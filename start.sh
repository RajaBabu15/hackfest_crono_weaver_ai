#!/bin/bash
set -e

echo "Starting Pathway Pipeline..."
python -m src.pathway_pipeline &
PW_PID=$!
echo "Pathway Pipeline PID: $PW_PID"

echo "Starting FastAPI Server..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "FastAPI Server PID: $API_PID"

echo "Starting Streamlit UI..."
# Corrected path separator here:
python -m streamlit run src/ui.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none

echo "Streamlit exited. Stopping background processes..."
kill -SIGTERM $PW_PID $API_PID || true
wait $PW_PID || true
wait $API_PID || true
echo "Shutdown complete."