"""Supervised fine-tuning (full fine-tuning) of the base model on a subset of Smol-SmolTalk.

Usage:
    python scripts/sft.py --config recipes/sft.yaml
    python scripts/sft.py --config recipes/sft.yaml --num_train_samples 2000 --max_steps 20   # quick check
"""

import os
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer, TrlParser


@dataclass
class DataArguments:
    model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_name: str = "HuggingFaceTB/smol-smoltalk"
    num_train_samples: int = 100_000
    num_eval_samples: int = 2_000
    wandb_entity: str | None = None
    wandb_project: str = "small-llm-post-training"


def load_split(args: DataArguments, split: str, num_samples: int, seed: int, tokenizer, max_length: int):
    """Random subset of complete conversations: longer ones are dropped rather than truncated."""

    def fits(batch):
        texts = tokenizer.apply_chat_template(batch["messages"], tokenize=False)
        return [len(ids) <= max_length for ids in tokenizer(texts)["input_ids"]]

    dataset = load_dataset(args.dataset_name, split=split).shuffle(seed=seed)
    # About 1% of conversations are too long: filter a 10% margin, then keep exactly num_samples
    dataset = dataset.select(range(min(len(dataset), int(num_samples * 1.1))))
    dataset = dataset.filter(fits, batched=True)
    return dataset.select(range(num_samples)).select_columns(["messages"])


def main(data_args: DataArguments, training_args: SFTConfig) -> None:
    if data_args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", data_args.wandb_entity)
    os.environ.setdefault("WANDB_PROJECT", data_args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path)
    original_chat_template = tokenizer.chat_template

    split_kwargs = dict(seed=training_args.data_seed, tokenizer=tokenizer, max_length=training_args.max_length)
    train_dataset = load_split(data_args, "train", data_args.num_train_samples, **split_kwargs)
    eval_dataset = load_split(data_args, "test", data_args.num_eval_samples, **split_kwargs)

    trainer = SFTTrainer(
        model=data_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # text-only: the tokenizer, not the multimodal processor
    )
    trainer.train()
    trainer.evaluate()  # final held-out loss, used to compare learning rates

    # The training template renders the same text, but ship the original one with the checkpoint
    trainer.processing_class.chat_template = original_chat_template
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((DataArguments, SFTConfig))
    data_args, training_args = parser.parse_args_and_config()
    main(data_args, training_args)
