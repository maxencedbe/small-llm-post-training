# Small LLM post-training: SFT → DPO → GRPO

An end-to-end post-training pipeline for a small language model, **Qwen3.5-0.8B-Base**: supervised fine-tuning, then preference optimization (DPO), then reinforcement learning with a verifiable reward (GRPO). Every stage is evaluated with the same protocol, to measure what each one brings.

**In short:** SFT turns the base model into an assistant (+15 points on IFEval), DPO mostly changes the style of its answers, and GRPO raises GSM8K math accuracy from 50.1% to 62.8% (3 seeds).

Training curves and results: [W&B report](https://forge.coreweave.com/wandb/maxence-debes-ml/small-llm-post-training/reports/Small-LLM-post-training-SFT-DPO-GRPO-on-Qwen3-5-0-8B--VmlldzoxODE2OTY0OQ?accessToken=r5riquti027x2maufd2wa9tjrjqrbsqv9qhnmgoarpk8q4dqnxk8hlxoa84aga12)

## Results

Scores in %, all models evaluated with the same fixed protocol (see [Evaluation](#evaluation)).

| Model | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst. strict | IFEval prompt loose | IFEval inst. loose | MMLU |
|---|---|---|---|---|---|---|---|
| Base (Qwen3.5-0.8B-Base) | 16.9 | 52.9 | 30.7 | 42.7 | 32.5 | 44.2 | 47.0 |
| + SFT | 48.7 | 48.6 | **45.7** | **56.8** | 49.0 | 60.1 | 51.9 |
| + DPO | 50.1 | 50.1 | 44.0 | 55.3 | 50.3 | 61.4 | **52.4** |
| **+ GRPO** (mean ± std, 3 seeds) | **62.8 ± 0.4** | **62.8 ± 0.3** | 41.3 ± 1.2 | 54.0 ± 0.9 | **51.0 ± 1.8** | **61.6 ± 1.2** | 52.3 ± 0.1 |
| *GRPO with a format-only reward (control, 1 seed)* | *43.4* | *43.3* | *40.1* | *53.2* | *49.7* | *60.9* | *52.4* |
| *Reference: Qwen3.5-0.8B (official post-trained)* | *33.7* | *49.7* | *52.3* | *63.6* | *56.8* | *67.9* | *45.3* |

In our checks, evaluating the same model twice moved generative scores by up to about 1 point, so smaller differences should not be read as real effects.

## What each stage brings

### SFT: from base model to assistant

100k conversations from Smol-SmolTalk turn the base model into an assistant that follows instructions: **IFEval prompt-level accuracy goes from 30.7 to 45.7** (+15 points), the largest IFEval gain of the pipeline.

On GSM8K, the SFT model starts following the format of the 5 few-shot examples in the prompt: 99% of its answers end with `#### <number>`, against 27% for the base model, so the **strict score triples (16.9 → 48.7)**. Its answers are also a third shorter, and flexible accuracy drops by 4.3 points; this is consistent with shorter answers skipping steps, but question difficulty also drives answer length, so the link is not established.

MMLU rises by 4.9 points. SFT adds no knowledge: the likely reason is that our protocol presents MMLU questions as a conversation, which suits the SFT model better than the base model (see [Evaluation](#evaluation)).

### DPO: preferences, mostly reflected in style

On held-out UltraFeedback pairs, the DPO model gives a higher implicit reward to the preferred answer in **74% of cases** (reward accuracy). On the benchmarks, scores move by less than 2 points. The visible effect is on length: on IFEval, the median answer becomes **60% longer** (203 → 325 tokens), a known bias of preference data, where the preferred answer is often the longer one.

### GRPO: more accurate math reasoning

With a binary reward for a correct final answer, **GSM8K accuracy rises from 50.1 to 62.8**, and the three seeds agree within 0.9 point. During training, the share of correct sampled answers goes from about 43% at the start to 61–78% at the end, depending on the seed. On the test set, GRPO fixes 262 answers and breaks 101 (seed 42); in the answers we read, it mostly fixes cases where the DPO model skipped part of the problem:

> *Eliza earns $10/hour for 40 hours and 1.2× that for overtime; she worked 45 hours.*
> DPO adds the overtime rate once: 400 + 12 = **412** ✗. GRPO: "she worked for 5 hours (45 − 40 = 5), her overtime pay is 12 × 5 = $60", 400 + 60 = **460** ✓.

A control run checks that the gain needs the correctness signal. It rewards any answer ending with `#### <number>`, right or wrong. Since the model already used this format in 99% of its answers, the reward reaches 100% within the first quarter of training and stops carrying information. The control ends at 43.4 on GSM8K (−6.7): its answers are shorter than the DPO model's (median 108 vs 125 tokens) and, in the answers we read, skip steps more often.

GRPO trains on math only, and IFEval drops by about 2.7 points, in the control run as well. MMLU does not move.

### About the reference model

The official post-trained Qwen3.5-0.8B is shown for scale and as a check of the protocol (its IFEval score matches its model card, see [Evaluation](#evaluation)). It is not a fair head-to-head comparison:

- **Specialist vs generalist.** Our GRPO stage trains on GSM8K itself, while Qwen's post-training covers many tasks at once.
- **Thinking disabled.** Qwen3.5 is designed to reason before answering; its model card reports much higher scores in thinking mode (e.g. 59.5 vs 48.5 on MMLU-Redux). For a uniform protocol, we evaluate every model without it, which is not the reference's strongest setting on math.
- **Answer format.** The reference ends only 62% of its GSM8K answers with `#### <number>`, hence its low strict score (33.7); its flexible score (49.7) is the fairer number.

On IFEval, the reference stays 6 to 11 points ahead, which reflects a much larger instruction-tuning effort than our 100k conversations and 54k preference pairs. Its lower MMLU score is not interpreted: MMLU is scored on the probability of answering with a letter right away, a setting that may not suit a model trained to reason first.

## Method

```
Qwen3.5-0.8B-Base ──SFT──▶ ──DPO──▶ ──GRPO──▶ final model
                  100k chats  53.6k pairs  GSM8K train
```

All stages are **full fine-tuning** with [TRL](https://github.com/huggingface/trl), on one GPU rented on [Modal](https://modal.com). Every hyperparameter lives in a YAML recipe under `recipes/`.

| Stage | Data | Main settings | Compute |
|---|---|---|---|
| SFT | [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): 100k random conversations, longer ones (>2,048 tokens, ~1%) dropped | 1 epoch, lr 2e-5 (best held-out loss in a 1e-5 / 2e-5 / 5e-5 sweep), cosine schedule, batch 128, loss on assistant tokens only | H100, 98 min |
| DPO | [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized): 53.6k pairs (12% ties dropped) | standard sigmoid loss, β = 0.1, lr 5e-7, 1 epoch, batch 128 | H100, 108 min |
| GRPO | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) train split (7,473 problems), 0-shot prompt asking for a final `#### <number>` line | 8 completions per problem, 128 completions per update, 1 epoch (467 updates), lr 1e-6, no KL penalty, TRL's default `dapo` loss, generation with vLLM on the same GPU | H100, ~1 h per run |

SFT and DPO follow the recipe used for SmolLM2 (SFT on Smol-SmolTalk, then DPO on UltraFeedback). Smol-SmolTalk contains little math, which leaves room for GRPO to improve GSM8K.

A few implementation details:

- **Chat template for SFT.** TRL needs `{% generation %}` markers to compute the loss on assistant tokens only, and Qwen's template has none. `chat_templates/qwen3_5_training.jinja` adds them and nothing else; a test checks that it renders the exact same text as the original.
- **Reward.** The GRPO reward extracts the answer with the same regular expression as lm-eval's strict GSM8K metric, so the model is rewarded on what it is graded on. (The reward reads the last `####` line and the metric the first; no evaluated answer contains more than one.)
- **Speed.** Installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) halved the SFT step time, since transformers otherwise falls back to a pure PyTorch implementation of the Gated DeltaNet layers.

## Evaluation

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the vLLM backend, greedy decoding, chat template applied, few-shot examples given as conversation turns, thinking disabled. The protocol is fixed in `recipes/eval.yaml`.

| Benchmark | What it measures | Setting |
|---|---|---|
| GSM8K | grade-school math | 5-shot, generation capped at 1,024 tokens; *strict* requires `#### <number>`, *flexible* takes the last number |
| IFEval | following verifiable instructions | 0-shot |
| MMLU | knowledge | 5-shot, log-likelihood of the answer letter |

Two choices worth knowing when reading the scores:

- **GSM8K generation cap.** lm-eval's default of 256 tokens cut off a quarter of the answers of both starting models before their final result. At 1,024 tokens, only 2.8% still hit the cap, and almost all of them are repetition loops.
- **Chat template on MMLU.** It likely lowers the base model's score: 47.0 here against 50.3 reported in a base-model setting by [dQwen3.5](https://arxiv.org/abs/2609.20751). The same protocol is applied to every model.

Sanity check: the official post-trained model scores 52.3 on IFEval here, against 52.1 on its [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## Limitations

- **One seed for SFT, DPO and the control run**, three for GRPO. Differences of about 1 point are within noise.
- **Off-policy preference data.** UltraFeedback responses were written by other, larger models; pairs sampled from the model itself would likely help DPO more.
- **One math benchmark.** GRPO is trained on the GSM8K train split and evaluated on its test split; transfer to other math benchmarks is not measured.
- **Train/eval prompt mismatch.** GRPO trains with a 0-shot prompt, while evaluation uses 5 few-shot examples.

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
