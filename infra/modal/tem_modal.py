"""Modal cloud compute for this repo (GPU training, preprocessing, inference).

All state lives on one private Modal Volume, ``tem-axon-myelin``:

    /vol/data/raw/train-dataset/{train_samples,train_instance_labels}/   challenge data (uploaded once)
    /vol/nnUNet_raw, /vol/nnUNet_preprocessed, /vol/nnUNet_results      nnU-Net folders
    /vol/runs/                                                           everything else (predictions, logs)

The repo's code (src/, scripts/, baselines/, configs/) is mounted at /root/repo on every run, so code edits
need no image rebuild. Data, .venv, external/ and docs/ are never uploaded with the code.

Usage (from the repo root; see docs/notes/13-modal.md):

    modal run infra/modal/tem_modal.py                                  # GPU smoke check (~1 min)
    modal run infra/modal/tem_modal.py --action cpu --cmd "python scripts/..."
    modal run --detach infra/modal/tem_modal.py --action gpu --cmd "..."
    modal run --detach infra/modal/tem_modal.py --action train --dataset 501 --fold 0

GPU functions are preemptible on Modal. ``train`` retries and resumes with nnU-Net's ``--c``; use the
``*_ckpt10`` trainers (baselines/nnunet/trainers.py) so a preemption loses at most 10 epochs.
"""

import os
import subprocess
from pathlib import Path

import modal

# Inside the container this module sits at /root/tem_modal.py; local paths are only read when building locally.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/repo")
GPU = os.environ.get("TEM_MODAL_GPU", "L40S")  # 48 GB: fits ResEnc-L (needs 24 GB). See doc 13 for costs.
CPU = 8.0  # nnU-Net augmentation is CPU-bound (nnUNet_n_proc_DA)
MEMORY_MB = 32768
HOUR = 60 * 60

volume = modal.Volume.from_name("tem-axon-myelin", create_if_missing=True)
VOL = "/vol"

ENV = {
    "nnUNet_raw": f"{VOL}/nnUNet_raw",
    "nnUNet_preprocessed": f"{VOL}/nnUNet_preprocessed",
    "nnUNet_results": f"{VOL}/nnUNet_results",
    "nnUNet_n_proc_DA": "8",
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
common = dict(volumes={VOL: volume}, cpu=CPU, memory=MEMORY_MB)


def _sh(cmd: str) -> None:
    """Run a shell command in the repo, stream output, commit the Volume, and raise on failure."""
    print(f"$ {cmd}", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd="/root/repo")
    finally:
        volume.commit()


@app.function(gpu=GPU, timeout=10 * 60, **common)
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
    import nnunetv2  # noqa: F401
    from nnunetv2.training.nnUNetTrainer.variants.tem.tem_trainers import nnUNetTrainer_250epochs_ckpt10  # noqa: F401

    print("cellpose", cellpose.version, "| custom nnU-Net trainers importable")
    print("CPU cores", os.cpu_count())
    for sub in ("data/raw/train-dataset/train_samples", "data/raw/train-dataset/train_instance_labels"):
        p = Path(VOL) / sub
        print(sub, len(list(p.glob("*.tif"))) if p.exists() else "missing (run the upload step)")


@app.function(timeout=6 * HOUR, **common)
def shell_cpu(cmd: str) -> None:
    """CPU-only job, e.g. data conversion or nnUNetv2_plan_and_preprocess."""
    _sh(cmd)


@app.function(
    gpu=GPU, timeout=24 * HOUR, retries=modal.Retries(initial_delay=0.0, max_retries=5),
    single_use_containers=True, **common,
)
def shell_gpu(cmd: str) -> None:
    """GPU job. It is retried after preemption, so ``cmd`` must be safe to re-run (resume, don't restart)."""
    _sh(cmd)


@app.function(
    gpu=GPU, timeout=24 * HOUR, retries=modal.Retries(initial_delay=0.0, max_retries=10),
    single_use_containers=True, **common,
)
def train_nnunet(dataset: str, configuration: str, fold: str, trainer: str, plans: str) -> None:
    """nnU-Net training that resumes from the newest checkpoint after a preemption or retry."""
    volume.reload()
    results = Path(ENV["nnUNet_results"])
    done = any(results.glob(f"Dataset{dataset}_*/{trainer}__{plans}__{configuration}/fold_{fold}/checkpoint_final.pth"))
    if done:
        print("checkpoint_final.pth exists; nothing to do")
        return
    started = any(results.glob(f"Dataset{dataset}_*/{trainer}__{plans}__{configuration}/fold_{fold}/checkpoint_*.pth"))
    cont = " --c" if started else ""
    _sh(f"nnUNetv2_train {dataset} {configuration} {fold} -tr {trainer} -p {plans} --npz{cont}")


@app.local_entrypoint()
def main(
    action: str = "check",
    cmd: str = "",
    dataset: str = "501",
    configuration: str = "2d",
    fold: str = "0",
    trainer: str = "nnUNetTrainer_250epochs_ckpt10",
    plans: str = "nnUNetResEncUNetLPlans",
) -> None:
    if action == "check":
        gpu_check.remote()
    elif action == "cpu":
        shell_cpu.remote(cmd)
    elif action == "gpu":
        shell_gpu.remote(cmd)
    elif action == "train":
        train_nnunet.remote(dataset, configuration, fold, trainer, plans)
    else:
        raise SystemExit(f"unknown action {action!r}: use check | cpu | gpu | train")
