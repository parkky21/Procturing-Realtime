import queue
import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
import sys
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SAMPLE_RATE = 16000
PROCESS_INTERVAL = 5 # seconds (how often to run the pipeline)
MAX_WINDOW_DURATION = 10 # seconds
TOKEN = os.getenv("HF_TOKEN")

print("Loading pipeline...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=TOKEN
    )
except Exception as e:
    print(f"Error loading pipeline: {e}")
    sys.exit(1)

print(f"Pipeline loaded. Starting continuous recording (max window {MAX_WINDOW_DURATION}s)...")
print("Press Ctrl+C to stop.")

# Global queue to hold incoming audio chunks safely
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Make a copy of the current chunk and put it in the queue
    audio_queue.put(indata.copy())

def process_buffer(current_buffer):
    # Run pipeline on in-memory dictionary
    try:
        # Convert to typical torch input (channels, time)
        waveform = torch.from_numpy(current_buffer).float().T
        
        audio_in_memory = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        
        start_time = time.time()
        output = pipeline(audio_in_memory)
        processing_time = time.time() - start_time
        
        buffer_duration = len(current_buffer) / SAMPLE_RATE
        print(f"\n--- Window ({buffer_duration:.1f}s) | Processing Time: {processing_time:.3f}s ---")
        
        found_speech = False
        unique_speakers = set()
        for turn, speaker in output.speaker_diarization:
            found_speech = True
            unique_speakers.add(speaker)
            print(f"  {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
        
        if len(unique_speakers) > 2:
            print("\n  !!! ALERT: MULTIPLE SPEAKERS DETECTED !!!")

        if not found_speech:
            print("  (No speech detected)")
            
    except Exception as e:
        print(f"Error processing buffer: {e}")

try:
    # Start the InputStream with the callback
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        
        accumulated_audio = np.zeros((0, 1), dtype='float32')
        last_process_time = 0
        
        while True:
            # 1. Retrieve all available chunks from the queue
            while not audio_queue.empty():
                chunk = audio_queue.get()
                accumulated_audio = np.concatenate((accumulated_audio, chunk))
            
            # 2. Enforce max window size (15s)
            max_samples = int(MAX_WINDOW_DURATION * SAMPLE_RATE)
            if len(accumulated_audio) > max_samples:
                accumulated_audio = accumulated_audio[-max_samples:]
            
            # 3. Check if it's time to process (every 5 seconds)
            # We use the length of the buffer or system time to decide frequency.
            # Simple heuristic: Just sleep a bit to avoid busy loop, 
            # but here we want to process as fast as possible BUT not faster than useful.
            # Let's process every time we have new significant data, 
            # OR simply: process, then wait.
            
            current_time = time.time()
            if current_time - last_process_time >= PROCESS_INTERVAL:
                if len(accumulated_audio) > 0:
                    # Cloning buffer for processing so we don't block the logic (though pipeline is blocking anyway)
                    # Since pipeline is blocking, the callback will keep filling the queue.
                    # When we return, we'll pop everything from queue. Ideal.
                    print(" Processing...", end="", flush=True)
                    process_buffer(accumulated_audio)
                    last_process_time = time.time()
            
            # Short sleep to prevent tight loop burning CPU when queue is empty
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"\nAn error occurred: {e}")
