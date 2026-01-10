import time
import os
from audio_handler import AudioHandler
from dotenv import load_dotenv

load_dotenv()

def test_audio_handler():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Skipping test: DEEPGRAM_API_KEY not set.")
        return

    print("Initializing AudioHandler...")
    handler = AudioHandler(api_key=api_key)
    
    print("Starting Handler...")
    handler.start()
    
    # Run for 5 seconds
    print("Running for 5 seconds...")
    try:
        for i in range(5):
            time.sleep(1)
            if not handler.main_thread or not handler.main_thread.is_alive():
                 print(f"Tick {i+1}: Thread DIED!")
                 break
            print(f"Tick {i+1}: Alive")
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping Handler...")
        handler.stop()
        print("Done.")

if __name__ == "__main__":
    test_audio_handler()
