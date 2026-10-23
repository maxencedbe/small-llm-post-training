"""Direct Preference Optimization (full fine-tuning) of the SFT model on UltraFeedback.

Usage:
    python scripts/dpo.py --config recipes/dpo.yaml
    python scripts/dpo.py --config recipes/dpo.yaml --num_eval_samples 200 --max_steps 10   # quick check
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer, TrlParser


@dataclass
class ScriptArguments:
    model_name_or_path: str = "checkpoints/sft"
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    num_eval_samples: int = 2_000
    wandb_entity: str | None = None
    wandb_project: str = "small-llm-post-training"


def load_split(args: ScriptArguments, split: str, tokenizer, max_length: int):
    """Preference pairs with a real preference, whose prompt and both responses fit in max_length."""

    def to_pair(example):
        # Explicit prompt / response format: the prompt is the user turn, each response the assistant turn
        return {"prompt": example["chosen"][:1], "chosen": example["chosen"][1:], "rejected": example["rejected"][1:]}

    def fits(batch):
        lengths = []
        for key in ("chosen", "rejected"):
            conversations = [prompt + response for prompt, response in zip(batch["prompt"], batch[key])]
            texts = tokenizer.apply_chat_template(conversations, tokenize=False)
            lengths.append([len(ids) for ids in tokenizer(texts)["input_ids"]])
        return [max(pair) <= max_length for pair in zip(*lengths)]

    dataset = load_dataset(args.dataset_name, split=split)
    # About 12% of pairs have equal scores: they carry no preference, so they are dropped
    dataset = dataset.filter(lambda example: example["score_chosen"] > example["score_rejected"])
    dataset = dataset.map(to_pair, remove_columns=dataset.column_names)
    return dataset.filter(fits, batched=True)


def main(script_args: ScriptArguments, training_args: DPOConfig) -> None:
    if script_args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", script_args.wandb_entity)
    os.environ.setdefault("WANDB_PROJECT", script_args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    train_dataset = load_split(script_args, "train_prefs", tokenizer, training_args.max_length)
    eval_dataset = load_split(script_args, "test_prefs", tokenizer, training_args.max_length)
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), script_args.num_eval_samples)))

    # The reference model is a frozen copy of the starting model, created by the trainer
    trainer = DPOTrainer(
        model=script_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # text-only: the tokenizer, not the multimodal processor
    )
    trainer.train()
    trainer.evaluate()

    trainer.save_model(training_args.output_dir)
    # vLLM loads Qwen3.5 as a multimodal model and needs its image and video processor configs
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy(Path(script_args.model_name_or_path) / name, training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)
