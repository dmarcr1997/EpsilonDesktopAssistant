import pyttsx3
import threading

class TTS:
    def __init__(self, rate: int = 175):
        self.rate = rate
        self._lock = threading.Lock()

    def say(self, text: str):
        """Speak text. Thread-safe - creates a fresh engine for each call."""
        with self._lock:
            engine = None
            try:
                # Create a fresh engine for each TTS call to avoid state issues
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                raise Exception(f"TTS error: {e}")
            finally:
                # Clean up the engine
                if engine is not None:
                    try:
                        engine.stop()
                    except:
                        pass
                    try:
                        del engine
                    except:
                        pass