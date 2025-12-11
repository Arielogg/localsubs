#!/bin/bash
# Stop the realtime subtitle server

if [ -f .subtitle_server.pid ]; then
    PID=$(cat .subtitle_server.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill $PID
        rm .subtitle_server.pid
        echo "Server stopped!"
    else
        echo "Server is not running (stale PID file)"
        rm .subtitle_server.pid
    fi
else
    # Fallback: find and kill by process name
    if pgrep -f "python main.py" > /dev/null; then
        echo "Stopping server..."
        pkill -f "python main.py"
        echo "Server stopped!"
    else
        echo "Server is not running"
    fi
fi
