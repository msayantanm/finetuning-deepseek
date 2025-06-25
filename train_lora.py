import os, argparse, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="deepseek-ai/deepseek-llm-7b-base")
    p.add_argument("--train_file",  default="data/train.jsonl")
    p.add_argument("--output_dir",  default="checkpoints/bn-lora")
    p.add_argument("--num_epochs",  type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--use_4bit",    action="store_true", default = True)
    return p.parse_args()

def main():
    args = parse()
    bnb_cfg = BitsAndBytesConfig(load_in_4bit = args.use_4bit and CUDA_AVAILABLE,
               bnb_4bit_compute_dtype = torch.float16,
               bnb_4bit_quant_type = "nf4")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        offload_folder=None,
        torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32,
        quantization_config = bnb_cfg,
        use_safetensors = True
        )

    if args.use_4bit and CUDA_AVAILABLE:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    if CUDA_AVAILABLE:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} "
        f"| Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
    else:
        print("CUDA NOT available – training on CPU.")

    ds = load_dataset("json", data_files=args.train_file, split="train")
    ds = ds.shuffle(seed=42)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=1024)
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        fp16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
