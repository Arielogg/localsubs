"""
Real-Time French → English Subtitle System

Main application that:
1. Captures audio from microphone
2. Uses VAD to detect speech segments
3. Transcribes and translates French → English with Whisper
4. Broadcasts subtitles via WebSocket to browser overlay
"""

import asyncio
import threading
import queue
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import config

# Global state
audio_queue: queue.Queue = queue.Queue()
subtitle_queue: asyncio.Queue = None
connected_clients: set = set()
is_running = True


def list_audio_devices():
    """List available audio input devices."""
    print("\n" + "=" * 60)
    print("Available Audio Input Devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_marker = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_marker}")
    print("=" * 60 + "\n")


def load_vad_model():
    """Load Silero VAD model."""
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    return model, get_speech_timestamps


def load_whisper_model():
    """Load faster-whisper model."""
    print(f"Loading Whisper model '{config.WHISPER_MODEL}' on {config.WHISPER_DEVICE}...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE
    )
    print("Whisper model loaded successfully!")
    return model


class AudioProcessor:
    """Handles audio capture and speech detection."""
    
    def __init__(self, vad_model, get_speech_timestamps, whisper_model):
        self.vad_model = vad_model
        self.get_speech_timestamps = get_speech_timestamps
        self.whisper_model = whisper_model
        
        self.audio_buffer = deque(maxlen=int(config.SAMPLE_RATE * config.MAX_CHUNK_DURATION_S))
        self.is_speech_active = False
        self.silence_start = None
        self.speech_start = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - runs in separate thread."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if stereo and flatten
        audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio_queue.put(audio_data.copy())
    
    def process_audio_chunk_streaming(self, audio_np: np.ndarray, subtitle_q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Process accumulated audio through Whisper and stream segments in real-time."""
        if len(audio_np) < config.SAMPLE_RATE * 0.5:  # Less than 0.5s
            return

        # Normalize audio
        audio_np = audio_np.astype(np.float32)
        if np.abs(audio_np).max() > 0:
            audio_np = audio_np / np.abs(audio_np).max()

        try:
            # Transcribe and translate French → English
            segments, info = self.whisper_model.transcribe(
                audio_np,
                language="fr",
                task="translate",  # This translates to English!
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                ),
                word_timestamps=config.WORD_BY_WORD_MODE,  # Enable word timestamps for word-by-word mode
            )

            # Choose streaming mode based on configuration
            if config.WORD_BY_WORD_MODE:
                # Word-by-word streaming mode
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        # Group words into n-grams
                        words_list = [w.word.strip() for w in segment.words]
                        for i in range(0, len(words_list), config.WORDS_PER_SUBTITLE):
                            word_group = words_list[i:i + config.WORDS_PER_SUBTITLE]
                            text = ' '.join(word_group)
                            if text:
                                print(f"[Subtitle] {text}")
                                asyncio.run_coroutine_threadsafe(
                                    subtitle_q.put(text),
                                    loop
                                )
                    else:
                        # Fallback if word timestamps not available
                        text = segment.text.strip()
                        if text:
                            print(f"[Subtitle] {text}")
                            asyncio.run_coroutine_threadsafe(
                                subtitle_q.put(text),
                                loop
                            )
            else:
                # Normal segment streaming mode
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        # Split long segments into lines if needed
                        if len(text) > config.SUBTITLE_MAX_CHARS:
                            words = text.split()
                            lines = []
                            current_line = []
                            current_length = 0

                            for word in words:
                                word_length = len(word) + 1  # +1 for space
                                if current_length + word_length > config.SUBTITLE_MAX_CHARS and current_line:
                                    lines.append(' '.join(current_line))
                                    current_line = [word]
                                    current_length = word_length
                                else:
                                    current_line.append(word)
                                    current_length += word_length

                            if current_line:
                                lines.append(' '.join(current_line))

                            # Send each line
                            for line in lines:
                                print(f"[Subtitle] {line}")
                                asyncio.run_coroutine_threadsafe(
                                    subtitle_q.put(line),
                                    loop
                                )
                        else:
                            print(f"[Subtitle] {text}")
                            asyncio.run_coroutine_threadsafe(
                                subtitle_q.put(text),
                                loop
                            )

        except Exception as e:
            print(f"Transcription error: {e}")
    
    def run_processing_loop(self, subtitle_q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Main processing loop - runs in separate thread."""
        print("Audio processing loop started")
        
        accumulated_audio = []
        last_speech_time = time.time()
        
        while is_running:
            try:
                # Get audio from queue with timeout
                audio_chunk = audio_queue.get(timeout=0.1)
                accumulated_audio.extend(audio_chunk)
                
                # Convert to numpy for VAD check
                audio_np = np.array(accumulated_audio, dtype=np.float32)
                
                # Simple energy-based speech detection as backup
                energy = np.sqrt(np.mean(audio_np[-len(audio_chunk):]**2))
                is_speech = energy > 0.01  # Adjust threshold as needed
                
                current_time = time.time()
                
                if is_speech:
                    last_speech_time = current_time
                
                # Check if we should process (silence detected or max duration reached)
                silence_duration = current_time - last_speech_time
                audio_duration = len(accumulated_audio) / config.SAMPLE_RATE
                
                should_process = (
                    (silence_duration > config.MIN_SILENCE_DURATION_MS / 1000 and audio_duration > 0.5) or
                    audio_duration >= config.MAX_CHUNK_DURATION_S
                )
                
                if should_process and len(accumulated_audio) > config.SAMPLE_RATE * 0.5:
                    # Process the accumulated audio with streaming segments
                    audio_to_process = np.array(accumulated_audio, dtype=np.float32)
                    accumulated_audio = []  # Reset buffer

                    # Stream transcription segments in real-time
                    self.process_audio_chunk_streaming(audio_to_process, subtitle_q, loop)
                
                # Prevent buffer from growing too large
                max_samples = int(config.SAMPLE_RATE * config.MAX_CHUNK_DURATION_S * 1.5)
                if len(accumulated_audio) > max_samples:
                    accumulated_audio = accumulated_audio[-max_samples:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        print("Audio processing loop ended")


# HTML template for subtitle overlay
OVERLAY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Subtitles</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: transparent;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }
        
        /* Toggle between transparent and black background with 'B' key */
        body.black-bg {
            background: #000;
        }
        
        #subtitle-container {
            padding: 20px 40px 60px;
            text-align: center;
        }
        
        #subtitle {
            display: inline-block;
            background: rgba(0, 0, 0, 0.85);
            color: #fff;
            font-size: 2.5rem;
            font-weight: 500;
            padding: 16px 32px;
            border-radius: 8px;
            max-width: 90%;
            line-height: 1.4;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            opacity: 1;
            transition: opacity 0.3s ease;
        }
        
        #subtitle.hidden {
            opacity: 0;
        }
        
        #subtitle:empty {
            display: none;
        }
        
        #status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        #status.connected {
            background: rgba(34, 197, 94, 0.9);
            color: white;
        }
        
        #status.disconnected {
            background: rgba(239, 68, 68, 0.9);
            color: white;
        }
        
        #instructions {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            color: #999;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.8rem;
            line-height: 1.5;
        }
        
        #instructions kbd {
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div id="status" class="disconnected">Disconnected</div>
    
    <div id="instructions">
        <kbd>F11</kbd> Fullscreen<br>
        <kbd>B</kbd> Toggle background<br>
        <kbd>+</kbd>/<kbd>-</kbd> Font size
    </div>
    
    <div id="subtitle-container">
        <div id="subtitle"></div>
    </div>
    
    <script>
        const subtitle = document.getElementById('subtitle');
        const status = document.getElementById('status');
        const instructions = document.getElementById('instructions');
        
        let ws = null;
        let hideTimeout = null;
        let currentFontSize = 2.5;
        const DISPLAY_TIME = 5000; // ms
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.className = 'connected';
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const text = event.data;
                showSubtitle(text);
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected';
                status.className = 'disconnected';
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connect, 2000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function showSubtitle(text) {
            // Clear any existing hide timeout
            if (hideTimeout) {
                clearTimeout(hideTimeout);
            }
            
            // Show subtitle
            subtitle.classList.remove('hidden');
            subtitle.textContent = text;
            
            // Auto-hide after delay
            hideTimeout = setTimeout(() => {
                subtitle.classList.add('hidden');
            }, DISPLAY_TIME);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key.toLowerCase()) {
                case 'b':
                    document.body.classList.toggle('black-bg');
                    break;
                case '+':
                case '=':
                    currentFontSize = Math.min(currentFontSize + 0.25, 5);
                    subtitle.style.fontSize = `${currentFontSize}rem`;
                    break;
                case '-':
                    currentFontSize = Math.max(currentFontSize - 0.25, 1);
                    subtitle.style.fontSize = `${currentFontSize}rem`;
                    break;
                case 'h':
                    instructions.style.display = instructions.style.display === 'none' ? 'block' : 'none';
                    break;
            }
        });
        
        // Auto-hide instructions after 10 seconds
        setTimeout(() => {
            instructions.style.opacity = '0';
            instructions.style.transition = 'opacity 1s';
            setTimeout(() => instructions.style.display = 'none', 1000);
        }, 10000);
        
        // Start connection
        connect();
    </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global subtitle_queue, is_running
    
    # Startup
    subtitle_queue = asyncio.Queue()
    
    # List audio devices
    list_audio_devices()
    
    # Load models
    vad_model, get_speech_timestamps = load_vad_model()
    whisper_model = load_whisper_model()
    
    # Create audio processor
    processor = AudioProcessor(vad_model, get_speech_timestamps, whisper_model)
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    
    # Start audio processing thread
    processing_thread = threading.Thread(
        target=processor.run_processing_loop,
        args=(subtitle_queue, loop),
        daemon=True
    )
    processing_thread.start()
    
    # Start audio stream
    device_index = config.AUDIO_DEVICE_INDEX if config.AUDIO_DEVICE_INDEX >= 0 else None
    
    print(f"\nStarting audio capture from device: {device_index or 'default'}")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz")
    print("\n" + "=" * 60)
    print("SERVER READY - Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    
    stream = sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=config.SAMPLE_RATE,
        callback=processor.audio_callback,
        blocksize=int(config.SAMPLE_RATE * 0.1),  # 100ms blocks
    )
    stream.start()
    
    # Start subtitle broadcast task
    broadcast_task = asyncio.create_task(broadcast_subtitles())
    
    yield
    
    # Shutdown
    is_running = False
    stream.stop()
    stream.close()
    broadcast_task.cancel()


app = FastAPI(lifespan=lifespan)


async def broadcast_subtitles():
    """Broadcast subtitles to all connected WebSocket clients."""
    while True:
        try:
            text = await subtitle_queue.get()
            
            # Send to all connected clients
            disconnected = set()
            for client in connected_clients:
                try:
                    await client.send_text(text)
                except:
                    disconnected.add(client)
            
            # Remove disconnected clients
            connected_clients.difference_update(disconnected)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Broadcast error: {e}")


@app.get("/")
async def get_overlay():
    """Serve the subtitle overlay page."""
    return HTMLResponse(content=OVERLAY_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time subtitle updates."""
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info"
    )
