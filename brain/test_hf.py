from .hf_engine import HFBrain, HFBrainConfig
from audio.mic import MicRecorder
from audio.stt_vosk import VoskSTT
from audio.tts import TTS
import os
from dotenv import load_dotenv
import time

load_dotenv()

tts_model = TTS()
mic = MicRecorder()
stt = VoskSTT(os.getenv("VOSK_MODEL_PATH"))
if __name__ == "__main__":
    brain = HFBrain(HFBrainConfig())
    print("Recording in 1 second... speak for ~3 seconds.")
    time.sleep(1)

    print("Starting recording...")
    mic.start()
    time.sleep(3)
    print("Stopping recording...")
    audio = mic.stop()

    text = stt.transcribe(audio)
    if text:
        reply = brain.reply(text)
        tts_model.say(reply);

