import time
import os
import json
import sys
import threading
import keyboard
from rich.console import Console

from brain.hf_engine import HFBrain, HFBrainConfig
from audio.mic import MicRecorder
from audio.stt_vosk import VoskSTT
from audio.tts import TTS
from dotenv import load_dotenv

load_dotenv()

console = Console()

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")

SYSTEM_PROMPT = """
You are Epsilon, a retro-futuristic maker assistant. You have the following params users can change:
(Humor: 50%, Conciseness: 20%, Professional: 40%).
respond in JSON like the following:
{
    "response": "text...."
}
"""

def main():
    brain = HFBrain(HFBrainConfig())
    mic = MicRecorder()
    stt = VoskSTT(VOSK_MODEL_PATH)
    tts = TTS()

    console.print("[bold cyan]EPSILON TALK LOOP ONLINE[/bold cyan]")
    console.print("Hold [bold]Enter[/bold] to record, release to stop. Press [bold]Esc[/bold] to exit.\n")

    recording = False
    should_exit = False
    processing = False
    
    def on_enter_press():
        nonlocal recording
        if not recording and not processing:
            recording = True
            console.print("[yellow]Recording...[/yellow]", end="\r")
            mic.start()
    
    def on_enter_release():
        nonlocal recording, processing
        if recording:
            recording = False
            console.print(" " * 50, end="\r")
            console.print("[yellow]Processing...[/yellow]", end="\r")
            
            audio = mic.stop()
            processing = True
            
            try:
                user_text = stt.transcribe(audio)
                console.print(" " * 50, end="\r")  # Clear processing line
                
                if not user_text:
                    console.print("[red]No speech detected.[/red]\n")
                    return

                console.print(f"[bold]You:[/bold] {user_text}")

                reply = json.loads(brain.reply(user_text, system=SYSTEM_PROMPT))["response"]
                console.print(f"[bold magenta]Epsilon:[/bold magenta] {reply}\n")

                def speak():
                    try:
                        tts.say(reply)
                    except Exception as e:
                        console.print(f"[red]TTS error: {e}[/red]")
                
                tts_thread = threading.Thread(target=speak, daemon=True)
                tts_thread.start()
                
            except Exception as e:
                console.print(f"[red]Processing error: {e}[/red]")
            finally:
                processing = False
                console.print("Ready. Hold Enter to talk...\n")
    
    def on_esc_press(event):
        nonlocal should_exit
        console.print("\n[yellow]Exiting...[/yellow]")
        should_exit = True
        return False
    
    # Register keyboard callbacks
    keyboard.on_press_key('enter', lambda _: on_enter_press())
    keyboard.on_release_key('enter', lambda _: on_enter_release())
    keyboard.on_press_key('esc', on_esc_press)
    
    console.print("Ready. Hold Enter to talk...\n")
    
    try:
        while True:
            if should_exit:
                break
            time.sleep(0.1) 

            if keyboard.is_pressed('esc'):
                should_exit = True
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted...[/yellow]")
        should_exit = True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        should_exit = True
    finally:
        console.print("[yellow]Cleaning up...[/yellow]")

        if recording:
            mic.stop()
        try:
            keyboard.unhook_all()
        except:
            pass  # Ignore errors during cleanup
        console.print("[bold cyan]EPSILON OFFLINE[/bold cyan]")

if __name__ == "__main__":
    main()