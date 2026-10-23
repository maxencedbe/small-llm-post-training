# Small LLM post-training: SFT → DPO → GRPO

A complete post-training pipeline for a small language model, **Qwen3.5-0.8B-Base**, with an evaluation after every stage. The project asks one question about the last stage: **when reinforcement learning with a verifiable reward improves math accuracy, does the gain come from better reasoning or just from a better answer format?**

**Main result.** GRPO with a correctness reward raises GSM8K accuracy from **50.1% to 62.8%** (mean of 3 seeds, ±0.4). A control run rewarded for the answer format alone *loses* 6.7 points. Since the starting model already used the expected format in 99% of its answers, the gain comes from more accurate reasoning, not from formatting.

Training curves and results: [W&B report](https://forge.coreweave.com/wandb/maxence-debes-ml/small-llm-post-training/reports/Small-LLM-post-training-SFT-DPO-GRPO-on-Qwen3-5-0-8B--VmlldzoxODE2OTY0OQ?accessToken=r5riquti027x2maufd2wa9tjrjqrbsqv9qhnmgoarpk8q4dqnxk8hlxoa84aga12)

## Results

All models are evaluated with the same fixed protocol (see [Evaluation](#evaluation)). Scores in %.

| Model | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst. strict | IFEval prompt loose | IFEval inst. loose | MMLU |
|---|---|---|---|---|---|---|---|
| Base (Qwen3.5-0.8B-Base) | 16.9 | 52.9 | 30.7 | 42.7 | 32.5 | 44.2 | 47.0 |
| + SFT | 48.7 | 48.6 | **45.7** | **56.8** | 49.0 | 60.1 | 51.9 |
| + DPO | 50.1 | 50.1 | 44.0 | 55.3 | 50.3 | 61.4 | **52.4** |
| **+ GRPO, correctness reward** (3 seeds) | **62.8 ± 0.4** | **62.8 ± 0.3** | 41.3 ± 1.2 | 54.0 ± 0.9 | **51.0 ± 1.8** | **61.6 ± 1.2** | 52.3 ± 0.1 |
| + GRPO, format-only reward (control) | 43.4 | 43.3 | 40.1 | 53.2 | 49.7 | 60.9 | 52.4 |
| *Reference: Qwen3.5-0.8B (official post-trained)* | *33.7* | *49.7* | *52.3* | *63.6* | *56.8* | *67.9* | *45.3* |

Re-running the same evaluation twice moves generative scores by about 1 point, so smaller differences should not be read as real effects.

## Findings

**1. GRPO improves reasoning, not format.** With a binary correctness reward, GSM8K rises by 12.6 points, and the three seeds agree within 0.9 point. The format-only control (reward 1 for any `#### <number>` line, right or wrong) saturates at 100% within the first quarter of training: from then on it carries no learning signal, its answers drift longer during training (from 230 to 490 tokens), and GSM8K drops by 6.7 points. On individual questions, GRPO mostly fixes answers where the DPO model skipped part of the problem:

> *Eliza earns $10/hour for 40 hours and 1.2× that for overtime; she worked 45 hours.*
> DPO adds the overtime rate once: 400 + 12 = **412** ✗. GRPO: "she worked for 5 hours (45 − 40 = 5), her overtime pay is 12 × 5 = $60", 400 + 60 = **460** ✓.

Overall, GRPO fixes 262 answers and breaks 101 (seed 42). Its answers are slightly longer (median 139 vs 125 tokens) and less tied to the terse style of the few-shot examples.

**2. SFT teaches the format.** After SFT, the model imitates the few-shot examples of the GSM8K prompt: 99% of answers end with `#### <number>` (27% for the base model), so the strict score triples (16.9 → 48.7). Its answers are also a third shorter, and flexible accuracy drops by 4.3 points. Shorter answers skip steps more often, but this link is not established: question difficulty also drives answer length. SFT is also where most of the IFEval gain happens (+15 points).

**3. DPO mostly lengthens answers.** On IFEval, DPO makes the median answer 60% longer (203 → 325 tokens), a known effect of preference data where the preferred answer is often the longer one. Scores barely move (all within 2 points): strict IFEval goes down, loose IFEval goes up.

**4. Side effects of GRPO.** IFEval drops by about 2.7 points after GRPO, in the format-only control as well: the cost comes from training on math alone, not from the correctness reward. MMLU stays flat, so no knowledge is forgotten.

**5. The model beats the official post-trained version on math, not on instructions.** After GRPO, it scores 62.8 on GSM8K against 49.7 for Qwen's own post-trained model (with thinking disabled), but stays 10 points behind it on IFEval.

## Method

```
Qwen3.5-0.8B-Base ──SFT──▶ ──DPO──▶ ──GRPO──▶ final model
                  100k chats  53.6k pairs  GSM8K train
```

All stages are **full fine-tuning** with [TRL](https://github.com/huggingface/trl), on one GPU rented on [Modal](https://modal.com). Every hyperparameter lives in a YAML recipe under `recipes/`.

| Stage | Data | Main settings | Compute |
|---|---|---|---|
| SFT | [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): 100k random conversations, longer ones (>2,048 tokens, ~1%) dropped | 1 epoch, lr 2e-5 (best of a 1e-5 / 2e-5 / 5e-5 sweep on held-out loss), cosine, batch 128, loss on assistant tokens only | H100, 98 min |
| DPO | [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized): 53.6k pairs (12% ties dropped) | standard sigmoid loss, β = 0.1, lr 5e-7, 1 epoch, batch 128 | H100, 108 min |
| GRPO | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) train split (7,473 problems), 0-shot prompt asking for a final `#### <number>` line | 8 completions per problem, 128 completions per update, 1 epoch (467 updates), lr 1e-6, no KL penalty, TRL's default `dapo` loss, generation with vLLM on the same GPU | H100, ~1 h per run |

SFT and DPO follow the recipe used for SmolLM2 (SFT on Smol-SmolTalk, then DPO on UltraFeedback). Smol-SmolTalk contains little math, which leaves the reasoning gain to GRPO and keeps it measurable.

A few details that mattered:

- **Chat template for SFT.** TRL needs `{% generation %}` markers to compute the loss on assistant tokens only, and Qwen's template has none. `chat_templates/qwen3_5_training.jinja` adds them and nothing else; a test checks that it renders the exact same text as the original.
- **Shared answer parser.** The GRPO reward extracts the answer with the same regular expression as lm-eval's strict GSM8K metric, so the model is rewarded on exactly what it is graded on.
- **Speed.** Installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) halved the SFT step time, since transformers otherwise falls back to a pure PyTorch implementation of the Gated DeltaNet layers.

## Evaluation

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the vLLM backend, greedy decoding, chat template applied, few-shot examples given as conversation turns, thinking disabled. The protocol is fixed in `recipes/eval.yaml`.

| Benchmark | What it measures | Setting |
|---|---|---|
| GSM8K | grade-school math | 5-shot, generation capped at 1,024 tokens; *strict* requires `#### <number>`, *flexible* takes the last number |
| IFEval | following verifiable instructions | 0-shot |
| MMLU | knowledge | 5-shot, log-likelihood of the answer letter |

Two pitfalls found while setting it up:

- **The default 256-token cap of lm-eval's GSM8K task cut off a quarter of the answers** of both starting models before their final result. At 1,024 tokens, only 2.8% still hit the cap, and almost all of them are repetition loops.
- **The chat template lowers the base model's MMLU score**: 47.0 here against 50.3 reported in a base-model setting by [dQwen3.5](https://arxiv.org/abs/2609.20751). This choice is applied to every model; it explains why MMLU rises after SFT.

Sanity check: the official post-trained model scores 52.3 on IFEval here, against 52.1 on its [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## Limitations

- **One seed for SFT and DPO**, three for GRPO. Differences of about 1 point are within noise.
- **Off-policy preference data.** UltraFeedback responses were written by other, larger models; pairs sampled from the model itself would likely help DPO more.
- **One reasoning benchmark.** The GRPO gain is measured on GSM8K only, the task it was trained on (on the test split).
- **Train/eval prompt mismatch.** GRPO trains with a 0-shot prompt; evaluation uses 5 few-shot examples. The gain transfers, but the model is not evaluated in the setting it was trained in.

## Reproducing

Requirements: [uv](https://docs.astral.sh/uv/), a [Modal](https://modal.com) account, and two Modal secrets, `wandb-secret` (`WANDB_API_KEY`) and `huggingface-secret` (`HF_TOKEN`).

```bash
uv sync
uv run modal setup
```

`modal_app.py` runs any script on a Modal GPU with the exact environment from `uv.lock`. Checkpoints are stored in a Modal volume.

```bash
# Evaluate a model (here the base model)
uv run modal run --detach modal_app.py --gpu L4 scripts/evaluate.py --config recipes/eval.yaml --model Qwen/Qwen3.5-0.8B-Base --run_name base

# Train each stage
uv run modal run --detach modal_app.py --gpu H100 scripts/sft.py --config recipes/sft.yaml
uv run modal run --detach modal_app.py --gpu H100 scripts/dpo.py --config recipes/dpo.yaml
uv run modal run --detach modal_app.py --gpu H100 scripts/grpo.py --config recipes/grpo.yaml --seed 42
uv run modal run --detach modal_app.py --gpu H100 scripts/grpo.py --config recipes/grpo.yaml --reward format --run_name grpo-format --output_dir checkpoints/grpo-format

# Evaluate a trained model
uv run modal run --detach modal_app.py --gpu L4 scripts/evaluate.py --config recipes/eval.yaml --model checkpoints/grpo-seed42 --run_name grpo-seed42
```

The scripts are plain Python and also run on a local NVIDIA GPU (`python scripts/sft.py --config recipes/sft.yaml`). Tests: `uv run pytest`.

## Repository layout

```
scripts/         evaluate.py, sft.py, dpo.py, grpo.py
recipes/         one YAML file per stage, with every hyperparameter
chat_templates/  Qwen3.5 template with {% generation %} markers for SFT
tests/           chat template and reward tests
modal_app.py     runs any script on a Modal GPU
```

## License

MIT
