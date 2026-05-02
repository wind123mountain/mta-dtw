import argparse
import json
import os
from typing import Optional
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

from data_utils.prompt_datasets import PromptDataset


def infer_model_type_from_path(model_path: str) -> str:
    lower = model_path.lower()
    if "gpt2" in lower:
        return "gpt2"
    if "tinyllama" in lower or "llama" in lower:
        return "tinyllama"
    if "mistral" in lower:
        return "mistral"
    return "gpt2"


def build_args(
    model_type: str,
    data_dir: str,
    json_data: bool,
    max_length: int,
    max_prompt_length: int,
) -> object:
    class Args:
        pass

    args = Args()
    args.model_type = model_type
    args.data_dir = data_dir
    args.json_data = json_data
    args.max_length = max_length
    args.max_prompt_length = max_prompt_length
    args.min_prompt_length = 1
    args.bin_data = False
    return args


def generate(
    model_path: Optional[str],
    base_model_path: Optional[str],
    lora_adapter_path: Optional[str],
    data_dir: str,
    split: str,
    save_path: Optional[str],
    batch_size: int,
    max_length: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    device: str,
):
    # Decide loading mode: full model vs base+LoRA
    use_lora = lora_adapter_path is not None
    model_source_path = base_model_path if use_lora else model_path
    if model_source_path is None:
        raise ValueError("Either --model-path or (--base-model-path and --lora-adapter-path) must be provided")

    tokenizer = AutoTokenizer.from_pretrained(model_source_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.to(device)
    model.eval()

    args = build_args(
        model_type=infer_model_type_from_path(model_source_path),
        data_dir=data_dir,
        json_data=True,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    dataset = PromptDataset(args, tokenizer, split, data_path=data_dir, num=-1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)

    generation_config = GenerationConfig(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

    outputs = []
    with torch.no_grad():
        for model_batch, no_model_batch in tqdm(dataloader, desc=f"Generating {split}"):
            dataset.move_to_device(model_batch, no_model_batch, device)
            query_ids = model_batch["input_ids"]
            attn_mask = model_batch["attention_mask"]

            max_new_tokens = max(1, max_length - query_ids.size(1))
            gen_out = model.generate(
                input_ids=query_ids,
                attention_mask=attn_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):]

            decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            for text in decoded:
                outputs.append({"text": text.strip()})

    if save_path is None:
        model_dir = lora_adapter_path if use_lora else model_path
        save_name = "answers_dolly.jsonl" if split == "dolly_100_samples" else f"answers_{split}.jsonl"
        save_path = os.path.join(model_dir, save_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return save_path, len(outputs)


def main():
    parser = argparse.ArgumentParser(description="Generate answers for dolly_100_samples.jsonl using a given model checkpoint")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the full fine-tuned model directory")
    parser.add_argument("--base-model-path", type=str, default=None, help="Base model path (e.g., TinyLLaMA/Mistral) for LoRA evaluation")
    parser.add_argument("--lora-adapter-path", type=str, default=None, help="LoRA adapter checkpoint directory (epochX_step...)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dolly"), help="Directory containing dataset jsonl files")
    parser.add_argument("--split", type=str, default="dolly_100_samples", help="Dataset split filename stem (without .jsonl)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to write answers jsonl; defaults to <model_path>/answers_dolly.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    save_path, n = generate(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        data_dir=args.data_dir,
        split=args.split,
        save_path=args.save_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        device=device,
    )
    print(f"Wrote {n} answers to {save_path}")


if __name__ == "__main__":
    main()