#!/bin/bash
# Start the realtime subtitle server in the background

# Check if already running
if pgrep -f "python main.py" > /dev/null; then
    echo "Server is already running!"
    echo "Use ./stop.sh to stop it first"
    exit 1
fi

echo "Starting realtime subtitle server..."

# Set library path and run in background
export LD_LIBRARY_PATH="./venv/lib/python3.12/site-packages/nvidia/cudnn/lib:./venv/lib/python3.12/site-packages/nvidia/cublas/lib:./venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# Run in background and save PID
nohup ./venv/bin/python main.py > subtitle_server.log 2>&1 &
echo $! > .subtitle_server.pid

echo "Server started! PID: $(cat .subtitle_server.pid)"
echo "View logs: tail -f subtitle_server.log"
echo "Stop server: ./stop.sh"
echo ""
echo "Open http://localhost:8000 in your browser"
