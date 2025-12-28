import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

@dataclass
class HFBrainConfig:
    model_id: str = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    device: str = os.getenv("HF_DEVICE", "auto")
    max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("HF_TEMPERATURE", "0.7"))

class HFBrain:
    def __init__(self, cfg: HFBrainConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        kwargs = {}
        if cfg.device == "auto":
            kwargs["device_map"] = "auto"
        elif cfg.device == "cpu":
            kwargs["device_map"] = {"": "cpu"}
        else:
            # e.g. "cuda"
            kwargs["device_map"] = {"": cfg.device}

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs
        )

        self.model.eval()
    
    @torch.inference_mode()
    def reply(self, user_text: str, system: str = "You are Epsilon, a retro-futuristic maker assistant. You have the following params users can change(Humor: 50%, Conciseness: 20%, Professional: 40%)") -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\nUser: {user_text}\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available() and self.cfg.device in ("auto", "cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]
        
        out = self.model.generate(
            **inputs,
            max_new_tokens = self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_tokens = out[0][input_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return text.strip()
