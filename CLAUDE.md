# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time French-to-English subtitle system for live events. Captures French audio from a microphone and displays translated English subtitles in a browser overlay, designed for live presentations and conferences.

## Development Commands

### Setup
```bash
# Create virtual environment
python3 -m venv venv

# Install dependencies
./venv/bin/pip install -r requirements.txt
```

### Testing Audio Devices
```bash
# List available audio devices and test microphone
./venv/bin/python test_audio.py
```
Use this to find the correct `AUDIO_DEVICE_INDEX` for your microphone.

### Running the Application

**IMPORTANT**: Use `./run.sh` instead of running `python main.py` directly. The run script sets up the CUDA library paths required for the venv's PyTorch to find cuDNN libraries.

```bash
# Foreground mode (see output, Ctrl+C to stop)
./run.sh

# Background mode (daemon)
./start.sh              # Start server in background
tail -f subtitle_server.log  # View logs
./stop.sh               # Stop server
```

Then open `http://localhost:8000` in a browser for the subtitle overlay.

### Configuration

Create a `.env` file (copy from `.env.example`) or set environment variables:

**Audio Settings:**
- `AUDIO_DEVICE_INDEX`: Microphone index (-1 for default)

**Whisper Model Settings:**
- `WHISPER_MODEL`: "tiny", "base", "small", "medium", "large-v3" (default: "small")
  - tiny: ~1GB VRAM, fastest, lowest quality
  - small: ~2GB VRAM, good balance (recommended)
  - medium: ~5GB VRAM, better quality
  - large-v3: ~10GB VRAM, best quality
- `WHISPER_DEVICE`: "cuda" or "cpu" (default: "cuda")
- `WHISPER_COMPUTE_TYPE`: "float16" (GPU) or "int8" (less VRAM)

**VAD Settings:**
- `MIN_SILENCE_DURATION_MS`: Silence threshold before processing (default: 500ms)
- `MIN_SPEECH_DURATION_MS`: Minimum speech duration to process (default: 250ms)
- `MAX_CHUNK_DURATION_S`: Force processing after this duration (default: 15s)

**Display Settings:**
- `SUBTITLE_MAX_CHARS`: Max characters per subtitle line (default: 120)
- `SUBTITLE_DISPLAY_TIME_S`: How long subtitles stay visible (default: 5.0s)

**Server Settings:**
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

**Word-by-Word Streaming Mode:**
- `WORD_BY_WORD_MODE`: Enable ultra-responsive word-by-word streaming (default: false)
- `WORDS_PER_SUBTITLE`: Number of words to group together (default: 3, range: 1-5)

To enable word-by-word mode, set `WORD_BY_WORD_MODE=true` in `.env` and adjust timing parameters:
```bash
WORD_BY_WORD_MODE=true
WORDS_PER_SUBTITLE=3
MIN_SILENCE_DURATION_MS=150
MAX_CHUNK_DURATION_S=2
```

You can also set environment variables inline:
```bash
WHISPER_MODEL=medium PORT=9000 ./run.sh
```

## Architecture

### Processing Pipeline
1. **Audio Capture** (`sounddevice`): Continuous microphone input at 16kHz
2. **Speech Detection** (Silero VAD): Energy-based detection to segment speech
3. **Translation** (`faster-whisper`): Transcribes French and translates to English in one step
4. **Broadcasting** (WebSocket): Real-time delivery to connected browser clients

### Threading Model
- **Main Thread**: FastAPI/Uvicorn server, WebSocket connections, async event loop
- **Audio Callback Thread**: Captures raw audio from microphone (via `sounddevice`)
- **Processing Thread**: Consumes audio from queue, performs VAD, runs Whisper inference, sends to async subtitle queue

### Key Components

**`main.py`**
- `AudioProcessor` class: Manages the audio capture → VAD → Whisper pipeline
- `audio_callback()`: Receives raw audio frames, queues them
- `run_processing_loop()`: Main processing loop that accumulates audio, detects silence, triggers Whisper transcription
- `process_audio_chunk()`: Runs Whisper translation (French→English with `task="translate"`)
- `broadcast_subtitles()`: Async task that forwards subtitles to all WebSocket clients
- `lifespan()`: Application startup/shutdown, loads models, starts threads

**Communication Flow**
- Microphone → `audio_callback` → `audio_queue` (thread-safe)
- Processing thread → Whisper → `subtitle_queue` (asyncio queue via `run_coroutine_threadsafe`)
- `broadcast_subtitles` → WebSocket clients

**`config.py`**
- Centralized configuration with environment variable fallbacks
- All settings documented with VRAM requirements

**`test_audio.py`**
- Interactive tool to list audio devices and test microphone levels
- Essential for troubleshooting audio input issues

### Hardware Requirements
- NVIDIA GPU recommended (6GB VRAM for medium model, 2GB for small)
- CUDA toolkit must be installed
- Runs on CPU as fallback (much slower)

**CUDA Library Path Issue:**
PyTorch ships with its own CUDA libraries (cuDNN, cublas, etc.) in the venv. The system's dynamic linker needs to find these libraries via `LD_LIBRARY_PATH`. The `run.sh` script handles this automatically by setting:
```bash
export LD_LIBRARY_PATH="./venv/lib/python3.12/site-packages/nvidia/cudnn/lib:./venv/lib/python3.12/site-packages/nvidia/cublas/lib:./venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
```
This only affects the running process and won't interfere with other projects or system CUDA installations.

### Whisper Translation Details
The Whisper model is configured with:
- `language="fr"`: Source language is French
- `task="translate"`: Direct translation to English (not transcription)
- `vad_filter=True`: Whisper's internal VAD filters out silence
- This is a single-step process: no separate transcription + translation

### Browser Overlay Features
- WebSocket auto-reconnect on connection loss
- Keyboard shortcuts: F11 (fullscreen), B (toggle background), +/- (font size), H (hide instructions)
- Auto-hide subtitles after 5 seconds (configurable)
- Connection status indicator
