"""Evaluate a model with lm-eval-harness, following the fixed protocol in recipes/eval.yaml.

Scores and per-sample generations are logged to W&B.

Usage:
    python scripts/evaluate.py --config recipes/eval.yaml --model Qwen/Qwen3.5-0.8B-Base --run_name base
    python scripts/evaluate.py --config recipes/eval.yaml --model checkpoints/sft --run_name sft
    python scripts/evaluate.py --config recipes/eval.yaml --model ... --run_name ... --limit 20   # quick check
"""

from dataclasses import dataclass, field

import wandb
from lm_eval import simple_evaluate
from lm_eval.loggers import WandbLogger
from lm_eval.models.vllm_causallms import VLLM
from trl import TrlParser


@dataclass
class EvalArguments:
    model: str = field(metadata={"help": "Hugging Face model id or local checkpoint path."})
    run_name: str = field(metadata={"help": "Name of the run in W&B, e.g. 'base', 'sft', 'grpo'."})
    tasks: list[str] = field(default_factory=lambda: ["gsm8k", "ifeval", "mmlu"])
    num_fewshot: dict[str, int] = field(default_factory=dict)
    max_gen_toks: dict[str, int] = field(default_factory=dict, metadata={"help": "Per-task generation length cap."})
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = True
    enable_thinking: bool = False
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    seed: int = 42
    limit: int | None = field(default=None, metadata={"help": "Examples per task, for quick checks only."})
    wandb_entity: str | None = None
    wandb_project: str = "small-llm-post-training"


def main(args: EvalArguments) -> None:
    # Start the W&B run first: if logging is misconfigured, fail before spending minutes loading the model
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"eval-{args.run_name}",
        job_type="eval",
        config=vars(args),
    )
    wandb_logger = WandbLogger()  # attaches to the run above

    # Load the model once with vLLM, then reuse it for every task
    lm = VLLM(
        pretrained=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )

    # Each task runs with its own few-shot setting, then all results are merged
    # and logged to W&B in one go (W&B does not allow overwriting the task configs)
    merged: dict = {}
    summary = {}
    for task in args.tasks:
        gen_kwargs = {"temperature": 0.0, "do_sample": False}  # greedy: same score on every rerun
        if task in args.max_gen_toks:
            gen_kwargs["max_gen_toks"] = args.max_gen_toks[task]
        results = simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=args.num_fewshot.get(task),
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            limit=args.limit,
            log_samples=True,
            random_seed=args.seed,
        )
        for key, value in results.items():
            if isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged.setdefault(key, value)

        scores = results.get("groups", {}).get(task) or results["results"][task]
        for metric, value in scores.items():
            if isinstance(value, float) and "stderr" not in metric:
                summary[f"{task}/{metric}"] = value

    wandb_logger.post_init(merged)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(merged["samples"])
    wandb.finish()

    print(f"\nResults for {args.run_name} ({args.model})")
    for name, value in summary.items():
        print(f"  {name:<45} {100 * value:6.2f}")


if __name__ == "__main__":
    parser = TrlParser(EvalArguments)
    (eval_args,) = parser.parse_args_and_config()
    main(eval_args)
