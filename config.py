"""
Configuration for the real-time subtitle system.
Override these via environment variables or by editing this file.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", -1))  # -1 = system default

# Whisper model settings
# Options: "tiny", "base", "small", "medium", "large-v3"
# VRAM usage: tiny ~1GB, small ~2GB, medium ~5GB, large-v3 ~10GB
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # "small" is safer for 6GB, "medium" for better quality
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # "float16" for GPU, "int8" for less VRAM

# VAD settings
MIN_SILENCE_DURATION_MS = int(os.getenv("MIN_SILENCE_DURATION_MS", 500))  # Silence before processing
MIN_SPEECH_DURATION_MS = int(os.getenv("MIN_SPEECH_DURATION_MS", 250))  # Minimum speech to process
MAX_CHUNK_DURATION_S = float(os.getenv("MAX_CHUNK_DURATION_S", 15.0))  # Force process after this duration

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Display settings
SUBTITLE_MAX_CHARS = int(os.getenv("SUBTITLE_MAX_CHARS", 120))  # Max characters per subtitle line
SUBTITLE_DISPLAY_TIME_S = float(os.getenv("SUBTITLE_DISPLAY_TIME_S", 5.0))  # How long subtitles stay visible

# Streaming mode settings
WORD_BY_WORD_MODE = os.getenv("WORD_BY_WORD_MODE", "false").lower() == "true"  # Enable word-by-word streaming
WORDS_PER_SUBTITLE = int(os.getenv("WORDS_PER_SUBTITLE", 3))  # Number of words to group together in word-by-word mode
