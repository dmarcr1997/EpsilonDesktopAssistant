import time
import os
import numpy as np
from .mic import MicRecorder
from .stt_vosk import VoskSTT
from dotenv import load_dotenv
from .tts import TTS

load_dotenv()

if __name__ == "__main__":

    mic = MicRecorder()
    stt = VoskSTT(os.getenv("VOSK_MODEL_PATH"))

    print("Recording in 1 second... speak for ~3 seconds.")
    time.sleep(1)

    print("Starting recording...")
    mic.start()
    time.sleep(3)
    print("Stopping recording...")
    audio = mic.stop()

    text = stt.transcribe(audio)
    print(f"Transcribed text: '{text}'")
    
    tts = TTS()
    tts.say(f"You said...{text}")
    print("STT:", text)