import os
import threading
import time
from collections import deque
import logging
from dotenv import load_dotenv
import numpy as np
import torch
from pyannote.audio import Pipeline
import pyaudio

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Pyannote Configuration
TOKEN = os.getenv("HF_TOKEN")
PROCESS_INTERVAL = 5 # seconds
MAX_WINDOW_DURATION = 10 # seconds

# Global Pipeline Singleton
GLOBAL_PIPELINE = None
PIPELINE_LOCK = threading.Lock()

def get_pipeline():
    global GLOBAL_PIPELINE
    with PIPELINE_LOCK:
        if GLOBAL_PIPELINE is None:
            logger.info("Loading Pyannote Pipeline (CPU)...")
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=TOKEN
                )
                # Force CPU or let it default. 
                # Usually defaults to CPU if no CUDA. 
                # To be strict: pipeline.to(torch.device("cpu"))
                pipeline.to(torch.device("cpu"))
                GLOBAL_PIPELINE = pipeline
                logger.info("Pyannote Pipeline loaded.")
            except Exception as e:
                logger.error(f"Error loading Pyannote pipeline: {e}")
    return GLOBAL_PIPELINE

class AudioHandler:
    def __init__(self, api_key=None):
        # API key not used anymore but kept for signature compatibility
        self.audio_alerts = deque(maxlen=1)
        
        # State
        self.is_running = False
        self.process_thread = None
        self.mic_thread = None
        
        # Audio Buffer (Stores float32 samples)
        self.audio_buffer = np.zeros((0, 1), dtype='float32')
        self.buffer_lock = threading.Lock()
        
        # Diarization state
        self.last_process_time = 0

    def start(self, use_microphone=True):
        if self.is_running:
            return

        self.is_running = True
        
        # Ensure pipeline is loaded
        get_pipeline()
        
        # Start Processing Thread
        self.process_thread = threading.Thread(
            target=self._run_processing_loop, 
            daemon=True
        )
        self.process_thread.start()
        
        # Start Mic Capture Thread if requested
        if use_microphone:
            self.start_microphone()
        
        logger.info(f"Audio handler started (Pyannote). Mic: {use_microphone}")

    def start_microphone(self):
        """Starts the microphone capture thread if not already running."""
        if self.mic_thread and self.mic_thread.is_alive():
            return

        try:
            self.mic_thread = threading.Thread(
                target=self._stream_audio_input, 
                daemon=True
            )
            self.mic_thread.start()
            logger.info("Microphone capture started.")
        except ImportError:
            logger.warning("PyAudio not available for microphone capture.")

    def stop(self):
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            self.process_thread = None
            
        if self.mic_thread:
            self.mic_thread.join(timeout=2.0)
            self.mic_thread = None
        
        logger.info("Audio handler stopped.")

    def get_latest_alerts(self):
        """Return list of new alerts since last call."""
        alerts = []
        while self.audio_alerts:
            alerts.append(self.audio_alerts.popleft())
        return alerts
    
    def process_audio_chunk(self, data: bytes):
        """
        Push external audio data (PCM bytes) to the buffer.
        """
        if not self.is_running:
            return
            
        try:
            # 1. Convert bytes (int16) to float32 numpy array
            # Assume 16-bit PCM, 16kHz
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Reshape to (N, 1) for pyannote (channels last or logic handles it)
            # Pyannote expects (channels, time) usually for tensor, but we store as (time, channels) for concatenation convenience
            chunk = audio_float32.reshape(-1, 1)
            
            with self.buffer_lock:
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                
                # Enforce max window size immediately to save memory? 
                # Or do it in processing loop. Doing it here prevents infinite growth between checks.
                max_samples = int(MAX_WINDOW_DURATION * RATE)
                if len(self.audio_buffer) > max_samples:
                    self.audio_buffer = self.audio_buffer[-max_samples:]
                    
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def _run_processing_loop(self):
        """
        Periodically runs diarization on the buffer.
        """
        while self.is_running:
            current_time = time.time()
            
            if current_time - self.last_process_time >= PROCESS_INTERVAL:
                
                # Get a snapshot of the buffer
                with self.buffer_lock:
                    if len(self.audio_buffer) < RATE * 1: # Need at least 1 second
                        time.sleep(0.5)
                        continue
                    current_window = self.audio_buffer.copy()
                
                self._analyze_audio(current_window)
                self.last_process_time = time.time()
            
            time.sleep(0.5)

    def _analyze_audio(self, buffer):
        pipeline = get_pipeline()
        if not pipeline:
            return

        try:
            # Prepare input for pipeline {"waveform": (channels, time), "sample_rate": 16000}
            # Buffer is (time, channels), so Transpose it.
            waveform = torch.from_numpy(buffer).float().T
            audio_in_memory = {"waveform": waveform, "sample_rate": RATE}
            
            # logger.info(f"Running diarization on {len(buffer)/RATE:.1f}s audio...")
            output = pipeline(audio_in_memory)
            
            unique_speakers = set()
            for turn, speaker in output.speaker_diarization:
                unique_speakers.add(speaker)
                # logger.debug(f"Detected {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
            
            if len(unique_speakers) > 1:
                self._trigger_alert("Multiple speakers detected!")
                
        except Exception as e:
            logger.error(f"Error in diarization: {e}")

    def _trigger_alert(self, msg):
        # Avoid spamming if the last alert was the same (simple debounce)
        # Note: deque maxlen=1 so this just checks the one sitting there or we append new.
        if not self.audio_alerts or self.audio_alerts[-1] != msg:
            self.audio_alerts.append(msg)
            logger.warning(f"Audio Alert: {msg}")

    def _stream_audio_input(self):
        """
        Captures audio from PyAudio and feeds process_audio_chunk.
        """
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            logger.info("Microphone capture started.")
            
            while self.is_running:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    if data:
                        self.process_audio_chunk(data)
                except Exception as e:
                    logger.error(f"Mic read error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error opening mic stream: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
