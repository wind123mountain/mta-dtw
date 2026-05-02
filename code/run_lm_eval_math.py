#!/usr/bin/env python3
"""Thin wrapper around lm-evaluation-harness for math benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.pop(sys.path.index(str(SCRIPT_DIR)))

from lm_eval import evaluator  # type: ignore  # noqa: E402
from lm_eval.tasks import TaskManager  # type: ignore  # noqa: E402
from lm_eval.utils import make_table  # type: ignore  # noqa: E402


def _split_tasks(raw: List[str]) -> List[str]:
    tasks: List[str] = []
    for entry in raw:
        for piece in entry.split(","):
            item = piece.strip()
            if item:
                tasks.append(item)
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LM Evaluation Harness on math benchmarks (GSM8K + MATH)."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Absolute path to the HF checkpoint directory (e.g., .../epoch1_stepXXXX).",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer path. Defaults to --model-path.",
    )
    parser.add_argument(
        "--model",
        default="hf",
        help="lm-eval model backend. Defaults to Hugging Face (`hf`).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gsm8k_cot", "hendrycks_math"],
        help="List of lm-eval task names or comma-separated string. "
        "Defaults to GSM8K (CoT) + full MATH.",
    )
    parser.add_argument(
        "--task-dir",
        help="Optional extra task directory to register with the harness.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override default few-shot count for tasks that support it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Global batch size passed to lm-eval.",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=None,
        help="Optional per-device batch size override.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Optional maximum batch size for automatic search.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit samples per task. If <1 treats as percentage. "
        "Leave unset for full evaluation.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for stderr estimation (0 disables).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Precision to request from hf backend.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string forwarded to lm-eval / HF backend.",
    )
    parser.add_argument(
        "--revision",
        help="Optional model revision/commit (useful for hub checkpoints).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the HF loader.",
    )
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Enable Accelerate inference path inside lm-eval.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit via bitsandbytes.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit via bitsandbytes.",
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=None,
        help="Optional override for generation max tokens via gen_kwargs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional override for generation temperature via gen_kwargs.",
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to store lm-eval artifacts. "
        "Defaults to <checkpoint parent>/lm_eval/math/<checkpoint name>.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Instruct lm-eval to dump per-sample generations.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable lm-eval request caching.",
    )
    parser.add_argument(
        "--cache-path",
        help="Path to sqlite cache db for lm-eval requests.",
    )
    parser.add_argument(
        "--rewrite-cache",
        action="store_true",
        help="Rewrite cached requests if they already exist.",
    )
    parser.add_argument(
        "--delete-cache",
        action="store_true",
        help="Delete cached requests before running.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for python/numpy/torch/few-shot sampling.",
    )
    parser.add_argument(
        "--metadata",
        help="Optional JSON string passed as metadata to tasks.",
    )
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Choose at most one of --load-in-4bit or --load-in-8bit.")
    return args


def main() -> int:
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    tokenizer_path = (
        Path(args.tokenizer_path).expanduser().resolve()
        if args.tokenizer_path
        else model_path
    )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
    else:
        save_dir = model_path.parent / "lm_eval" / "math" / model_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    task_list = _split_tasks(args.tasks)
    if not task_list:
        raise ValueError("At least one task must be provided via --tasks.")

    model_args = [
        f"pretrained={model_path}",
        f"tokenizer={tokenizer_path}",
        f"dtype={args.dtype}",
    ]
    if args.revision:
        model_args.append(f"revision={args.revision}")
    if args.trust_remote_code:
        model_args.append("trust_remote_code=True")
    if args.use_accelerate:
        model_args.append("use_accelerate=True")
    if args.load_in_8bit:
        model_args.append("load_in_8bit=True")
    if args.load_in_4bit:
        model_args.append("load_in_4bit=True")

    gen_kwargs = {}
    if args.gen_max_tokens is not None:
        gen_kwargs["max_gen_toks"] = args.gen_max_tokens
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature

    metadata_obj = None
    if args.metadata:
        try:
            metadata_obj = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            raise ValueError("--metadata must be valid JSON.") from exc

    task_manager = TaskManager(include_path=args.task_dir, metadata=metadata_obj)

    eval_kwargs = dict(
        model=args.model,
        model_args=",".join(model_args),
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        cache_requests=not args.no_cache,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=args.log_samples,
        task_manager=task_manager,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    if args.device_batch_size is not None:
        eval_kwargs["device_batch_size"] = args.device_batch_size
    if args.cache_path:
        eval_kwargs["use_cache"] = args.cache_path
    if args.rewrite_cache:
        eval_kwargs["rewrite_requests_cache"] = True
    if args.delete_cache:
        eval_kwargs["delete_requests_cache"] = True

    if gen_kwargs:
        eval_kwargs["gen_kwargs"] = gen_kwargs

    if metadata_obj is not None:
        eval_kwargs["metadata"] = metadata_obj

    cmd_str = " ".join(shlex.quote(part) for part in sys.argv)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"[lm-eval] Running with command:\n{cmd_str}\n")
    results = evaluator.simple_evaluate(**eval_kwargs)

    results_path = save_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    table = make_table(results)
    summary_path = save_dir / "summary.txt"
    summary_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nResults stored under: {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

