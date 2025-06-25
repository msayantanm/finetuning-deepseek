#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load base + LoRA adapter and generate Bengali text.

Example:  python generate.py --adapter checkpoints/bn-lora
"""
import argparse, torch, textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

CUDA_AVAILABLE = torch.cuda.is_available()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="deepseek-ai/deepseek-llm-7b-base")
    p.add_argument("--adapter", required=True, help="Path to LoRA checkpoint dir")
    p.add_argument("--prompt", default="একটি কবিতা শুরু করি:")
    p.add_argument("--max_new_tokens", type=int, default=120)
    return p.parse_args()

def main():
    a = parse()
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        a.base,
        device_map="auto",
        torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    if CUDA_AVAILABLE:
        print(f"Inference on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT available – running on CPU.")

    inputs = tok(a.prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        temperature=0.9, top_p=0.95, do_sample=True,
        max_new_tokens=a.max_new_tokens,
        pad_token_id=tok.eos_token_id
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    print(textwrap.fill(tok.decode(out[0], skip_special_tokens=True), 100))

if __name__ == "__main__":
    main()
