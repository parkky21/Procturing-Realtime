import os
import threading
import time
from collections import deque
import logging
from dotenv import load_dotenv

import pyaudio

# Try importing deepgram; strictly needed for this functionality
try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
except ImportError:
    DeepgramClient = None
    print("Warning: deepgram-sdk or pyaudio not installed. Audio proctoring will fail.")

# Load environment variables
load_dotenv()

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.deepgram = None
        self.audio_alerts = deque(maxlen=1)
        
        # Helper state
        self.is_running = False
        self.main_thread = None
        self.dg_connection = None
        
        # Diarization state
        self.seen_speakers = set()

        if not self.api_key:
            print("Warning: DEEPGRAM_API_KEY not found. Audio proctoring will be disabled.")
            return

        if DeepgramClient:
            try:
                # Initialize SDK
                # Initialize SDK with explicit key
                self.deepgram = DeepgramClient(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Deepgram client: {e}")

    def start(self, use_microphone=True):
        if not self.deepgram:
            print("Deepgram client not initialized. Cannot start audio proctoring.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.seen_speakers.clear()
        
        # Start the main management thread
        self.main_thread = threading.Thread(
            target=self._run_deepgram_session, 
            args=(use_microphone,), 
            daemon=True
        )
        self.main_thread.start()
        print(f"Audio proctoring started (Deepgram nova-3). Mic: {use_microphone}")

    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        if self.main_thread:
            self.main_thread.join(timeout=3.0)
            self.main_thread = None
        
        print("Audio proctoring stopped.")

    def get_latest_alerts(self):
        """Return list of new alerts since last call."""
        alerts = []
        while self.audio_alerts:
            alerts.append(self.audio_alerts.popleft())
        return alerts
    
    def process_audio_chunk(self, data: bytes):
        """
        Push external audio data (e.g. from LiveKit) to Deepgram.
        """
        # Ensure connection exists and is ready
        if self.is_running and self.dg_connection and len(data) > 0:
             try:
                 # Depending on SDK version this might need verify. 
                 # 'send' or 'send_media' is used for raw bytes.
                 if hasattr(self.dg_connection, 'send_media'):
                    self.dg_connection.send_media(data)
                 elif hasattr(self.dg_connection, 'send'):
                    self.dg_connection.send(data)
             except Exception as e:
                 # print(f"Error sending audio chunk: {e}")
                 pass

    def _run_deepgram_session(self, use_microphone=True):
        """
        Manages the Deepgram connection lifecycle within a thread.
        Emulates the 'with connect(...) as connection' pattern.
        """
        try:
            # Connect using the new SDK V3 pattern options directly in connect, or via kwargs override
            # We want 'nova-3', 'diarize=True', 'smart_format=True'
            
            with self.deepgram.listen.v1.connect(
                model="nova-3",
                language="en-US",
                smart_format=True,
                diarize=True,
                encoding="linear16",
                channels=CHANNELS,
                sample_rate=RATE
            ) as connection:
                
                self.dg_connection = connection

                # --- Setup Callbacks ---
                def on_message(result, **kwargs):
                    self._process_transcript(result)

                def on_error(error, **kwargs):
                    print(f"Deepgram Error: {error}")

                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                
                # --- Start Listening ---
                if hasattr(connection, 'start_listening'):
                    listen_thread = threading.Thread(target=connection.start_listening, daemon=True)
                    listen_thread.start()

                # --- Start Audio Streaming Thread (Only if using Mic) ---
                if use_microphone:
                    stream_thread = threading.Thread(
                        target=self._stream_audio_input, 
                        args=(connection,), 
                        daemon=True
                    )
                    stream_thread.start()

                # --- KeepAlive Loop ---
                # To prevent 1011 limit if audio is silent/sparse
                def keep_alive():
                    logging.info("Deepgram KeepAlive loop started.")
                    last_ka = time.time()
                    while self.is_running and self.dg_connection:
                        # Send KeepAlive every 5 seconds
                        if time.time() - last_ka > 5:
                            try:
                                # SDK v3: 'send' might not exist. 'send_media' is clear.
                                # Send a silent audio frame as keepalive
                                if hasattr(connection, 'keep_alive'):
                                    connection.keep_alive()
                                else:
                                    # 20ms of silence at 16kHz mono 16-bit = 640 bytes
                                    # Just sending a small buffer.
                                    silent_frame = b'\x00' * 320
                                    # Default to send_media for data
                                    if hasattr(connection, 'send_media'):
                                        connection.send_media(silent_frame)
                                    else:
                                        # Fallback (very old or very new SDK?)
                                        pass
                                
                                # logging.debug("Sent KeepAlive (Silence)")
                                last_ka = time.time()
                            except Exception as e:
                                logging.error(f"KeepAlive failed: {e}")
                                break
                        time.sleep(1)
                
                ka_thread = threading.Thread(target=keep_alive, daemon=True)
                ka_thread.start()

                # --- Wait Loop ---
                # Keep this thread alive until stop() is called, keeping the 'with' block active
                while self.is_running:
                    time.sleep(0.1)
                
                self.dg_connection = None

        except Exception as e:
            print(f"Error in Deepgram session loop: {e}")
            self.is_running = False

    def _stream_audio_input(self, connection):
        """
        Captures audio from PyAudio and sends it to Deepgram.
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
            
            while self.is_running:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if data:
                    # Send media as per user example
                    connection.send_media(data)
                    
        except Exception as e:
            print(f"Error streaming audio: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def _process_transcript(self, message):
        """
        Process the transcript result to detect multiple speakers.
        """
        try:
            # Parse message. SDK v3 usually returns an object.
            # We access properties safely.
            
            channel = getattr(message, 'channel', None)
            if channel and hasattr(channel, 'alternatives'):
                alt = channel.alternatives[0]
                words = alt.words
                
                # words is a list of Word objects
                if words:
                    current_chunk_speakers = set()
                    for word in words:
                        # 'speaker' is the ID of the speaker
                        if hasattr(word, 'speaker'):
                            spk = word.speaker
                            self.seen_speakers.add(spk)
                            current_chunk_speakers.add(spk)
                    
                    # Logic 1: Simultaneous
                    if len(current_chunk_speakers) > 1:
                        self._trigger_alert("Multiple voices detected simultaneously!")
                        
                    # Logic 2: Cumulative
                    if len(self.seen_speakers) > 1:
                        self._trigger_alert("Multiple speakers detected in session!")
                            
        except Exception as e:
            # print(f"Error processing transcript: {e}") 
            pass

    def _trigger_alert(self, msg):
        # Avoid spamming
        if not self.audio_alerts or self.audio_alerts[-1] != msg:
            self.audio_alerts.append(msg)
            print(f"Audio Alert: {msg} (Speakers found: {self.seen_speakers})")
