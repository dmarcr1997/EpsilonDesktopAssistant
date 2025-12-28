import json
import numpy as np
from vosk import Model, KaldiRecognizer

class VoskSTT:
    def __init__(self, model_path: str, samplerate: int = 16000):
        self.model = Model(model_path)
        self.samplerate = samplerate
    
    def transcribe(self, audio_float32: np.ndarray) -> str:
        if len(audio_float32) == 0:
            return ""
        
        pcm16 = (audio_float32 * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
        
        rec = KaldiRecognizer(self.model, self.samplerate)
        
        chunk_size = 4000  # Process in ~0.25 second chunks
        total_chunks = len(pcm16) // chunk_size
        
        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i + chunk_size]
            if rec.AcceptWaveform(chunk):
                partial = json.loads(rec.Result())
                if partial.get("text"):
                    print(f"STT: Partial result: '{partial.get('text')}'")
        
        result = json.loads(rec.FinalResult())
        text = (result.get("text") or "").strip()
        if text == '':
            text = partial.get('text')
        print(f"STT: Final result: '{text}'")
        return text