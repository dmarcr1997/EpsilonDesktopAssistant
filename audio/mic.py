import queue
import sounddevice as sd
import numpy as np

class MicRecorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1, device: int = None):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._chunks = []
    
    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}")
        self._q.put(indata.copy())
    
    def start(self):
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    
    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        while not self._q.empty():
            self._chunks.append(self._q.get())


        if not self._chunks:
            return np.zeros((0,), dtype=np.float32)

        audio = np.concatenate(self._chunks, axis=0).squeeze()
        return audio