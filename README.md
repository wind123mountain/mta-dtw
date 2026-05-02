# Enviroment

- The environment requires: Python 3.10 or 3.11

- Install all dependencies via `pip install -r requirements.txt`


# Training

### Our method: DWA-KD
Use the following scripts to launch experiments:
- GPT2-base: `bash scripts/gpt2/dwa_kd_gpt2_base.sh`
- GPT2-medium: `bash scripts/gpt2_medium/dwa_kd_gpt2_medium.sh`
- GPT2-XL: `bash scripts/gpt2xl/dwa_kd_gpt2xl.sh`
- TinyLLaMA-1.1B: `bash scripts/tinyllama/dwa_kd_tinyllama.sh`
- OPT-2.7B: `bash scripts/opt/dwa_kd_opt.sh`

### SFT for teacher models
- Qwen1.5-1.8B (full fine-tuning): `bash scripts/gpt2/sft_teacher_qwen.sh`
- Mistral-7B (LoRA): `bash scripts/tinyllama/sft_teacher_mistral.sh`
- OPT-2.7B (LoRA): `bash scripts/opt/sft_teacher_opt.sh`

### SFT for student models
- GPT2-base (full fine-tuning): `bash scripts/gpt2/sft_gpt2_base.sh`
- GPT2-medium (full fine-tuning): `bash scripts/gpt2_medium/sft_gpt2_medium.sh`
- GPT2-XL (full fine-tuning): `bash scripts/gpt2xl/sft_gpt2xl.sh`
- TinyLLaMA-1.1B (LoRA): `bash scripts/tinyllama/sft_tinyllama.sh`
- OPT-2.7B (LoRA): `bash scripts/opt/sft_opt.sh`

### Baseline: Dual-Space KD with CMA
- GPT2-base: `bash scripts/gpt2/dskd_cma_gpt2_base.sh`
- GPT2-medium: `bash scripts/gpt2_medium/dskd_cma_gpt2_medium.sh`
- GPT2-XL: `bash scripts/gpt2xl/dskd_cma_gpt2xl.sh`
- TinyLLaMA-1.1B: `bash scripts/tinyllama/dskd_cma_tinyllama.sh`
- OPT-2.7B: `bash scripts/opt/dskd_cma_opt.sh`

### Baseline: Logits Alignment by Minimum Edit Distance 
- GPT2-base: `bash scripts/gpt2/minedit_gpt2_base.sh`
- GPT2-medium: `bash scripts/gpt2_medium/minedit_gpt2_medium.sh`
- GPT2-XL: `bash scripts/gpt2xl/minedit_gpt2xl.sh`
- TinyLLaMA-1.1B: `bash scripts/tinyllama/minedit_tinyllama.sh`
- OPT-2.7B: `bash scripts/opt/minedit_opt.sh`

### Baseline: Universal Logit Distillation 
- GPT2-base: `bash scripts/gpt2/uld_gpt2_base.sh`
- GPT2-medium: `bash scripts/gpt2_medium/uld_gpt2_medium.sh`
- GPT2-XL: `bash scripts/gpt2xl/uld_gpt2xl.sh`
- TinyLLaMA-1.1B: `bash scripts/tinyllama/uld_tinyllama.sh`
- OPT-2.7B: `bash scripts/opt/uld_opt.sh`

You can change the distance functions using `KD_OBJ` in the above scripts.


### File Structures in Output Directory
The output directory will be created under `./outputs` automatically after you run the training scripts. 
For full fine-tuning, the file structure of the output directory is as follows (take gpt2 SFT as an example):
```
./outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```
For LoRA fine-tuning, the file structure of the output directory is as follows (take TinyLLaMA LoRA SFT as an example):
```
./outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```



## Evaluation
### Evaluate Full Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```
According to the above structure, `CKPT_PATH` is the **absolute path** of the model files like `/home/xxx/DSKD/outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../epochA_step...`.

### Evaluate LoRA Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```
Please note that `MODEL_PATH` in `run_eval_lora.sh` should be changed for different base models (TinyLLaMA, LLaMA2, Mistral).

Similarly, `LORA_ADAPTER_PATH` is the **absolute path** of the LoRA adapter files like `/home/xxx/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../epochA_step...`.

### Evaluate Math Benchmarks (GSM8K + MATH) with LM Evaluation Harness

Install the extra dependency (`lm_eval`) via `pip install -r requirements.txt`, then run:

```bash
# Evaluate a Gemma-3-270M checkpoint on GSM8K (CoT) + full MATH
bash scripts/eval/run_lm_eval_math.sh /abs/path/to/epochX_stepXXXX [BATCH_SIZE] [LIMIT]
```

The helper script wraps `code/run_lm_eval_math.py`, which:
- loads the checkpoint/tokenizer with Hugging Face `AutoModelForCausalLM`
- runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks `gsm8k_cot` and `math`
- saves JSON + tabular summaries under `<checkpoint_parent>/lm_eval/math/<checkpoint_name>/`

Flags such as precision (`--dtype`), generation kwargs, or custom tasks can be passed by invoking the Python entrypoint directly; run `python code/run_lm_eval_math.py --help` for the complete list of options.


## Data
The processed data used in our paper can be downloaded [here](https://drive.google.com/drive/folders/1ZUsNVgWevACV9D-AHVNi9C7PX_2itzb8?usp=sharing).

## Models
You can download the corresponding model files (e.g., `pytorch_model.bin` or `model.safetensors`) of LLMs used in this paper into `model_hub/*/*/`.

Here are the links of these models on huggingface:
- GPT2-120M: [Here](https://huggingface.co/openai-community/gpt2)
- GPT2-1.5B (trained on Dolly by Gu et al.): [Here](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- Qwen1.5-1.8B: [Here](https://huggingface.co/Qwen/Qwen1.5-1.8B)
- TinyLLaMA-1.1B: [Here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Mistral-7B: [Here](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- OPT-2.7B: [Here](https://huggingface.co/facebook/opt-2.7b)