# Real-Time French → English Subtitle System

A real-time speech-to-text translation system for live events. Captures French audio and displays English subtitles via a browser overlay.

## Requirements

- Python 3.10+
- NVIDIA GPU with ~6GB VRAM (for `medium` model) or ~2GB (for `small` model)
- CUDA toolkit installed

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Open the subtitle overlay:**
   - Navigate to `http://localhost:8000` in a browser
   - Press F11 for fullscreen
   - Display this on your projector

3. **Configure audio input:**
   - The system will list available audio devices on startup
   - Set the correct device index in `config.py` or via environment variable

## Configuration

Edit `config.py` or set environment variables:

- `AUDIO_DEVICE_INDEX`: Microphone device index (default: system default)
- `WHISPER_MODEL`: Model size - "small", "medium", "large-v3" (default: "medium")
- `MIN_SILENCE_DURATION`: Seconds of silence before processing chunk (default: 0.5)
- `MAX_CHUNK_DURATION`: Maximum chunk duration in seconds (default: 10)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Microphone  │────▶│  Silero VAD │────▶│  faster-whisper  │────▶│  WebSocket  │
│   Input     │     │  (chunking) │     │   (translate)    │     │   Server    │
└─────────────┘     └─────────────┘     └──────────────────┘     └──────┬──────┘
                                                                        │
                                                                        ▼
                                                                 ┌─────────────┐
                                                                 │   Browser   │
                                                                 │   Overlay   │
                                                                 └─────────────┘
```

