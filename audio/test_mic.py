"""Simple microphone test to verify audio capture is working."""
import time
import numpy as np
from .mic import MicRecorder

if __name__ == "__main__":
    
    print("Testing microphone capture...")
    
    mic = MicRecorder()
    
    mic.start()
    print("Recording... (Press Ctrl+C to stop)\n")
    
    sample_count = 0
    while True:
        time.sleep(0.1)  # Check every 100ms
        
        chunks = []
        while not mic._q.empty():
            chunks.append(mic._q.get())
        
        if chunks:
            # Process the latest chunk
            latest = chunks[-1].squeeze()
            abs_max = np.max(np.abs(latest))
            rms = np.sqrt(np.mean(latest**2))
            
            # Visual indicator
            bar_length = int(abs_max * 50)  # Scale to 50 chars
            bar = "█" * min(bar_length, 50)
            
            print(f"\r[{sample_count:04d}] Max: {abs_max:.4f} | RMS: {rms:.4f} | {bar}", end="", flush=True)
            sample_count += 1
            
            # Put chunks back for proper recording
            for chunk in chunks:
                mic._q.put(chunk)
