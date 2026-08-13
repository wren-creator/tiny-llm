#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/app.pid"
LOG_FILE="$SCRIPT_DIR/app.log"
PORT="${PORT:-8501}"

# Check for an existing run via PID file
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "tiny-llm is already running (PID $OLD_PID). Use ./stop.sh to stop it first."
        exit 1
    else
        echo "Stale PID file found (PID $OLD_PID is gone). Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

# Also check by process name in case PID file is missing
EXISTING=$(pgrep -f "streamlit run.*app\.py" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Warning: found existing tiny-llm process(es) without a PID file: $EXISTING"
    echo "Run ./stop.sh to clean them up, or kill manually."
    exit 1
fi

cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "Starting tiny-llm..."
nohup streamlit run app.py --server.port "$PORT" --server.headless true >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "tiny-llm started (PID $(cat "$PID_FILE")) at http://localhost:$PORT — logs: $LOG_FILE"
