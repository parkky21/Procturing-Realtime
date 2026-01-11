# download the pipeline from Huggingface
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
from dotenv import load_dotenv

load_dotenv()
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN")
    )

# run the pipeline locally on your computer

with ProgressHook() as hook:
    output = pipeline("What is the best advice Alia Bhatt has ever received_ _jayshetty _podcast _aliabhatt.wav", hook=hook)  # runs locally

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
