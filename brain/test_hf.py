from hf_engine import HFBrain, HFBrainConfig

if __name__ == "__main__":
    brain = HFBrain(HFBrainConfig())
    print(brain.reply("I’m wiring a TB6612FNG to an ESP32. What are the common mistakes?"))
