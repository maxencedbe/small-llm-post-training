"""Run any project script on a Modal GPU, with the exact environment from uv.lock.

The scripts themselves stay plain Python: they run the same way on Modal or on a local NVIDIA GPU.

Usage:
    modal run modal_app.py --gpu L4 scripts/evaluate.py --config recipes/eval.yaml --model ... --run_name ...
    modal run --detach modal_app.py ...   # keep running if the laptop disconnects

Secrets expected in the Modal workspace: "wandb-secret" (WANDB_API_KEY) and "huggingface-secret" (HF_TOKEN).
"""

import subprocess

import modal

PROJECT_DIR = "/root/project"
CACHE_DIR = "/cache"
TIMEOUT = 24 * 60 * 60  # Modal's maximum; the default (5 min) is far too short for training

image = (
    # CUDA devel image: FlashInfer (used by vLLM) compiles some kernels with nvcc at runtime.
    # CUDA 13.0 matches the torch wheels in uv.lock.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_sync(extra_options="--no-dev")  # Modal provides its own client inside the container
    .add_local_dir(
        ".",
        PROJECT_DIR,
        # Mounted at startup, so code changes do not rebuild the image
        ignore=[".git", ".venv", "checkpoints", "outputs", "wandb", "**/__pycache__", ".DS_Store"],
    )
)

# Model weights and vLLM/Triton compilation caches, kept between runs
cache = modal.Volume.from_name("llm-post-training-cache", create_if_missing=True)
# Trained models, so that later runs (evaluation, DPO, GRPO) can load them
checkpoints = modal.Volume.from_name("llm-post-training-checkpoints", create_if_missing=True)

app = modal.App("small-llm-post-training", image=image)


@app.function(
    volumes={CACHE_DIR: cache, f"{PROJECT_DIR}/checkpoints": checkpoints},
    # Set at runtime only, so image builds never write into the volume path.
    # Hugging Face and vLLM follow XDG_CACHE_HOME; Triton and TileLang (FLA kernels) need their own variables
    env={
        "XDG_CACHE_HOME": CACHE_DIR,
        "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
        "TILELANG_CACHE_DIR": f"{CACHE_DIR}/tilelang",
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT,
)
def run(args: list[str]) -> None:
    subprocess.run(["python", *args], cwd=PROJECT_DIR, check=True)


@app.local_entrypoint()
def main(*args: str) -> None:
    # The first option picks the GPU, everything else is passed to the script unchanged
    gpu, args = (args[1], args[2:]) if args[:1] == ("--gpu",) else ("L4", args)
    # spawn() rather than remote(): with --detach, the call keeps running if the laptop disconnects
    run.with_options(gpu=gpu).spawn(list(args)).get()
