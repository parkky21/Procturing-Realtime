# Proctoring-AI Microservice

A scalable, real-time proctoring agent designed for integration with LiveKit and other streaming platforms. It uses advanced computer vision and audio analysis to monitor candidate behavior during online interviews or exams.

## Features

### 1. Vision Analysis (Deep Learning)
- **Eye Gaze Tracking**: Detects if the candidate is looking away (Left/Right/Up/Down).
- **Head Pose Estimation**: Monitors head orientation to ensure focus.
- **Object Detection (YOLOv11)**: Detects unauthorized usage of **Mobile Phones** or presence of **Multiple Persons**.
- **Face Spoofing**: Checks for liveness to prevent photo-based cheating.

### 2. Audio Analysis (Deepgram Nova-3)
- **Multi-Speaker Detection (Diarization)**: Alerts if more than one distinct speaker is detected in the session.
- **Speech-to-Text**: Real-time transcription (optional).

### 3. Microservice Architecture
- **Server**: FastAPI application exposing WebSocket endpoints for processing.
- **Client**: Lightweight adapter (`Procturing` class) compatible with **LiveKit Agents**.

---

## Architecture

The system is split into two components to facilitate scalability and isolation.

### 1. AI Server (`app/main.py`)
Exposes WebSocket endpoints where clients stream raw data. Each session operates in isolation.

**Endpoints per Session (`{session_id}`):**
- **`/ws/proctor/{session_id}/video`**: Accepts binary JPEG images.
- **`/ws/proctor/{session_id}/audio`**: Accepts raw PCM audio bytes (16kHz, 16-bit).
- **`/ws/proctor/{session_id}/events`**: Sends JSON alerts (e.g., `{"alerts": ["Looking Left", "Mobile Phone Detected"]}`).

### 2. LiveKit Client Adapter (`app/services/procturing.py`)
A python module designed to run within a LiveKit Worker.
- Consumes `rtc.VideoTrack` and `rtc.AudioTrack`.
- Forwards frame data to the AI Server via WebSockets.
- Receives alerts from the server.

---

## Installation

### Prerequisites
- Python 3.8+
- [Deepgram API Key](https://deepgram.com)

### MacOS Troubleshooting
If you encounter `Library not loaded` errors with `torchcodec` particularly related to `libavutil` or `ffmpeg`, you may need to specify the library path manually.

We have provided a helper script for this:
```bash
sh run_pipeline.sh
```

Or manually:
```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
uv run test.py
```

### Steps
1. Clone the repository.
2. Install dependencies (using `uv` is recommended for speed):
   ```bash
   pip install uv
   uv sync
   # OR standard pip
   pip install -r requirements.txt
   ```
### 3. Pyannote Audio (Speaker Diarization)
This project uses `pyannote.audio` for local speaker detection (replacing Deepgram).
- **Token Required**: You must have a HuggingFace token with access to `pyannote/speaker-diarization-community-1`.
- **Setup**: The token is currently hardcoded in `app/services/audio_handler.py` (variable `TOKEN`). Ensure this is valid.

### 4. Running the Application

### 1. Start the AI Server
```bash
uv run run.py
# Server will start on http://0.0.0.0:8000
```
This starts the processing server. It includes a local debugging UI at `http://localhost:8000`.

### 2. Integrate with LiveKit Agent
In your LiveKit Agent's `entrypoint.py`:

```python
from app.services.procturing import Procturing

# ... inside your track subscription handler ...
proctor = Procturing(participant, room)

if track.kind == rtc.TrackKind.KIND_VIDEO:
    proctor.video_track = track
    await proctor.start_if_ready()
    
if track.kind == rtc.TrackKind.KIND_AUDIO:
    await proctor.enable_audio(track)
```

The client will automatically connect to `ws://localhost:8000/ws/proctor/...` and start streaming.

---

## Performance
- **Throttling**: Vision analysis runs at optimized intervals to balance accuracy and compute cost.
    - **Face/Gaze**: Every 3 frames (~10 FPS).
    - **Object Detection**: Every 30 frames (~1 FPS).
- **Latency**: End-to-end processing latency is typically **<100ms** on local networks.

---

## Legacy/Local Mode
For standalone testing without LiveKit, the project retains a local camera mode:
1. Run `uv run run.py`.
2. Open `http://localhost:8000` in your browser.
3. Allow camera permissions.
