"""Modal cloud compute for this repo: **GPU work only**.

CPU work runs on the local Mac (M4):
- data conversion, nnU-Net planning and preprocessing
- pairing, evaluation and analysis

Modal is used only for training and inference, so we don't pay for cloud CPU-only containers. See
docs/notes/13-modal.md §3b for the full split and the GPU-seconds rules.

State lives on one private Modal Volume, ``tem-axon-myelin``:

    /vol/data/raw/train-dataset/                    challenge data (uploaded 12 Nov)
    /vol/nnUNet_preprocessed/Dataset<ID>_<name>/    preprocessed locally, then uploaded with `modal volume put`
    /vol/nnUNet_results/                            checkpoints, logs, validation softmax (.npz)
    /vol/runs/                                      GPU-utilisation logs, budget ledgers, predictions

Usage (from the repo root; see docs/notes/13-modal.md §5):

    modal run infra/modal/tem_modal.py                                      # GPU smoke check (~1 min)
    modal run infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_5epochs_ckpt1 --max-gpu-hours 0.3
    modal run --detach infra/modal/tem_modal.py --action train --dataset 501 --fold 0 --max-gpu-hours 5
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

# Inside the container this module sits at /root/tem_modal.py; local paths are only read when building locally.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/repo")
# L4 (24 GB) is the best value for a 2D U-Net on a small budget. ResEnc-M targets 8 GB of VRAM, and nnU-Net 2D
# training is largely CPU-bound (augmentation). Override with TEM_MODAL_GPU=A10G or L40S. See doc 13 §3a.
GPU = os.environ.get("TEM_MODAL_GPU", "L4")
# The GPU container's own CPU feeds the GPU with augmented batches during training, which can't move to the Mac
# (every batch would cross the network). Six cores is the smallest number that should keep an L4 busy. The GPU log
# shows whether that holds: below ~70% utilisation, the CPU is starving the GPU.
CPU = 6.0
MEMORY_MB = 16384
HOUR = 60 * 60

volume = modal.Volume.from_name("tem-axon-myelin", create_if_missing=True)
VOL = "/vol"
LOCAL_PRE = "/tmp/nnUNet_preprocessed"  # container-local SSD copy, so patch reads don't go over the network

ENV = {
    "nnUNet_raw": f"{VOL}/nnUNet_raw",
    "nnUNet_preprocessed": f"{VOL}/nnUNet_preprocessed",
    "nnUNet_results": f"{VOL}/nnUNet_results",
    "nnUNet_n_proc_DA": str(int(CPU)),
    "TEM_DATA": f"{VOL}/data/raw/train-dataset",
    "TEM_RUNS": f"{VOL}/runs",
    "PYTHONPATH": "/root/repo/src:/root/repo",
    "PYTHONUNBUFFERED": "1",
}

# Versions match the local .venv (docs/notes/12-day0-5-resources.md §2–3). PyPI's Linux torch wheels bundle CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.15.0",
        "nnunetv2==2.8.1",
        "cellpose==4.2.1.1",
        "monai==1.6.1",
        "scikit-image==0.26.0",
        "opencv-python-headless",
        "PyWavelets",
        "tifffile",
        "imageio",
        "pandas",
        "pyyaml",
        "tqdm",
        "axonpath @ git+https://github.com/paucabar/axonpath@6d633bf0407558b827cbcc430b72b59d5b5a6ec6",
    )
    # nnU-Net only finds trainers inside its own package, so install ours there.
    .add_local_file(REPO / "baselines/nnunet/trainers.py", "/tmp/tem_trainers.py", copy=True)
    .run_commands(
        "python -c \"import nnunetv2, os, shutil; d = os.path.join(os.path.dirname(nnunetv2.__file__), "
        "'training/nnUNetTrainer/variants/tem'); os.makedirs(d, exist_ok=True); "
        "open(os.path.join(d, '__init__.py'), 'w').close(); "
        "shutil.copy('/tmp/tem_trainers.py', os.path.join(d, 'tem_trainers.py'))\""
    )
    .env(ENV)
    .add_local_dir(
        REPO,
        "/root/repo",
        ignore=[
            ".git", ".venv", "data", "external", "docs", "notebooks", "submission",
            "**/__pycache__", "**/*.tif", "**/*.tiff", "**/*.zip", "**/*.pth", "**/*.pt",
            "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "runs", "outputs", "CLAUDE.md",
        ],
    )
)

app = modal.App("tem-axon-myelin", image=image)
gpu_fn = {
    "gpu": GPU, "cpu": CPU, "memory": MEMORY_MB, "volumes": {VOL: volume},
    "single_use_containers": True,  # a retry after preemption starts clean; nothing idles between calls
}


def _gpu_logger(path: Path) -> subprocess.Popen:
    """Sample GPU utilisation every 15 s. Low utilisation means paid GPU seconds are being wasted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")  # noqa: SIM115 (closed by the OS when the process ends)
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
         "-l", "15"], stdout=f, stderr=subprocess.DEVNULL,
    )


def _gpu_summary(path: Path) -> None:
    rows = [r.split(", ") for r in path.read_text().splitlines() if r.count(",") == 2] if path.exists() else []
    util = [float(r[1]) for r in rows if r[1].strip().isdigit()]
    if util:
        busy = sum(u >= 50 for u in util) / len(util)
        print(f"GPU utilisation: mean {sum(util) / len(util):.0f}%, ≥50% busy in {busy:.0%} of samples ({path})")


def _run_with_deadline(cmd: str, deadline: float, env: dict) -> int | None:
    """Run ``cmd``. If it is still running at ``deadline`` (epoch seconds), stop it. Returns the exit code, or None."""
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd="/root/repo", env=env, start_new_session=True)
    try:
        return proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        print("GPU-hour cap reached: stopping. checkpoint_latest is at most 10 epochs old.", flush=True)
        os.killpg(proc.pid, 15)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
        return None


def _ledger(key: str, max_gpu_hours: float) -> dict:
    """Per-run budget ledger on the Volume. The deadline is fixed at the first attempt, so retries can't extend it."""
    path = Path(VOL) / "runs" / "budget" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        led = json.loads(path.read_text())
    else:
        led = {"key": key, "first_start": time.time(), "max_gpu_hours": max_gpu_hours, "attempts": 0}
    led["attempts"] += 1
    led["deadline"] = led["first_start"] + led["max_gpu_hours"] * HOUR
    path.write_text(json.dumps(led, indent=1))
    volume.commit()
    return led


@app.function(timeout=10 * 60, **gpu_fn)
def gpu_check() -> None:
    """Report GPU, CUDA, library versions and what is on the Volume (T-00 pre-flight step 2)."""
    import torch

    subprocess.run(["nvidia-smi"], check=False)
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {p.name}, {p.total_memory / 2**30:.1f} GiB")
        x = torch.randn(4096, 4096, device="cuda")
        print("matmul ok:", float((x @ x).abs().mean()))
    import cellpose
    from nnunetv2.training.nnUNetTrainer.variants.tem.tem_trainers import (  # noqa: F401
        nnUNetTrainer_250epochs_ckpt10_es,
    )

    print("cellpose", cellpose.version, "| custom nnU-Net trainers importable")
    for sub in ("data/raw/train-dataset/train_samples", "data/raw/train-dataset/train_instance_labels"):
        p = Path(VOL) / sub
        print(sub, len(list(p.glob("*.tif"))) if p.exists() else "missing")
    pre = Path(ENV["nnUNet_preprocessed"])
    print("preprocessed datasets:", sorted(d.name for d in pre.iterdir()) if pre.exists() else "none yet")


@app.function(timeout=24 * HOUR, retries=modal.Retries(initial_delay=0.0, max_retries=3), **gpu_fn)
def train_nnunet(dataset: str, configuration: str, fold: str, trainer: str, plans: str, max_gpu_hours: float) -> None:
    """Resumable nnU-Net training with early stopping (trainer) and a hard GPU-hour cap (ledger)."""
    volume.reload()
    key = f"nnunet_D{dataset}_{trainer}_{plans}_{configuration}_f{fold}"
    led = _ledger(key, max_gpu_hours)
    if time.time() >= led["deadline"]:
        print(f"Budget ledger {key}: the GPU-hour cap ({led['max_gpu_hours']} h) is already used up. Not starting.")
        return

    results = Path(ENV["nnUNet_results"])
    run_dir = f"Dataset{dataset}_*/{trainer}__{plans}__{configuration}/fold_{fold}"
    if any(results.glob(f"{run_dir}/checkpoint_final.*")):
        print("checkpoint_final exists; nothing to do")
        return

    # Copy the preprocessed dataset to local SSD once per container (blosc2 files, no unpacking: seconds, not minutes).
    src = next(Path(ENV["nnUNet_preprocessed"]).glob(f"Dataset{dataset}_*"), None)
    if src is None:
        raise SystemExit(f"No preprocessed Dataset{dataset} on the Volume. Preprocess locally and upload it first.")
    t0 = time.time()
    shutil.copytree(src, f"{LOCAL_PRE}/{src.name}", dirs_exist_ok=True)
    print(f"staged {src.name} to local disk in {time.time() - t0:.0f} s")
    env = {**os.environ, "nnUNet_preprocessed": LOCAL_PRE}

    cont = " --c" if any(results.glob(f"{run_dir}/checkpoint_*")) else ""
    val_best = " --val_best" if trainer.endswith("_es") else ""
    gpu_log = Path(VOL) / "runs" / "gpu_util" / f"{key}.csv"
    sampler = _gpu_logger(gpu_log)
    try:
        code = _run_with_deadline(
            f"nnUNetv2_train {dataset} {configuration} {fold} -tr {trainer} -p {plans} --npz{cont}{val_best}",
            led["deadline"], env,
        )
    finally:
        sampler.terminate()
        _gpu_summary(gpu_log)
        volume.commit()
    if code not in (0, None):
        raise RuntimeError(f"nnUNetv2_train exited with {code}")  # lets Modal retry (resumes with --c)


@app.function(timeout=6 * HOUR, **gpu_fn)
def predict_nnunet(dataset: str, configuration: str, fold: str, trainer: str, plans: str,
                   input_dir: str, output_dir: str, checkpoint: str, save_probabilities: bool) -> None:
    """Sliding-window inference only. Input images (nnU-Net format) are prepared locally and uploaded first."""
    volume.reload()
    prob = " --save_probabilities" if save_probabilities else ""
    cmd = (f"nnUNetv2_predict -i {VOL}/{input_dir} -o {VOL}/{output_dir} -d {dataset} -c {configuration} "
           f"-f {fold} -tr {trainer} -p {plans} -chk {checkpoint}{prob}")
    try:
        code = _run_with_deadline(cmd, time.time() + 6 * HOUR, dict(os.environ))
    finally:
        volume.commit()
    if code != 0:
        raise RuntimeError(f"nnUNetv2_predict exited with {code}")


@app.function(timeout=24 * HOUR, retries=modal.Retries(initial_delay=0.0, max_retries=3), **gpu_fn)
def shell_gpu(cmd: str, max_gpu_hours: float) -> None:
    """Any other GPU job (e.g. Cellpose fine-tuning). It is retried after preemption, so ``cmd`` must resume."""
    volume.reload()
    try:
        code = _run_with_deadline(cmd, time.time() + max_gpu_hours * HOUR, dict(os.environ))
    finally:
        volume.commit()
    if code not in (0, None):
        raise RuntimeError(f"command exited with {code}")


@app.local_entrypoint()
def main(
    action: str = "check",
    cmd: str = "",
    dataset: str = "501",
    configuration: str = "2d",
    fold: str = "0",
    trainer: str = "nnUNetTrainer_250epochs_ckpt10_es",
    plans: str = "nnUNetResEncUNetMPlans",
    max_gpu_hours: float = 5.0,
    input_dir: str = "",
    output_dir: str = "",
    checkpoint: str = "checkpoint_best.pth",
    save_probabilities: bool = True,
) -> None:
    if action == "check":
        gpu_check.remote()
    elif action == "train":
        train_nnunet.remote(dataset, configuration, fold, trainer, plans, max_gpu_hours)
    elif action == "predict":
        predict_nnunet.remote(dataset, configuration, fold, trainer, plans, input_dir, output_dir, checkpoint,
                              save_probabilities)
    elif action == "gpu":
        shell_gpu.remote(cmd, max_gpu_hours)
    else:
        raise SystemExit(f"unknown action {action!r}: use check | train | predict | gpu (CPU work runs locally)")
