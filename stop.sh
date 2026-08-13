#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/app.pid"

stopped=0

# Stop via PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping tiny-llm (PID $PID)..."
        kill "$PID"
        # Wait up to 5 seconds for clean exit
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "Process did not exit cleanly; sending SIGKILL..."
            kill -9 "$PID"
        fi
        echo "Stopped."
        stopped=1
    else
        echo "Stale PID file (PID $PID is not running). Cleaning up."
    fi
    rm -f "$PID_FILE"
fi

# Also sweep for any orphaned processes not tracked by the PID file
ORPHANS=$(pgrep -f "streamlit run.*app\.py" 2>/dev/null || true)
if [ -n "$ORPHANS" ]; then
    echo "Found orphaned tiny-llm process(es): $ORPHANS — killing them."
    echo "$ORPHANS" | xargs kill 2>/dev/null || true
    stopped=1
fi

if [ "$stopped" -eq 0 ]; then
    echo "No running tiny-llm process found."
fi
