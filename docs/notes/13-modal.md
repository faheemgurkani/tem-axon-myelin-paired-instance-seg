# 13 — Cloud compute on Modal (12 Nov 2026)

Modal replaces the dedicated GPU machine planned in [12](12-day0-5-resources.md) §1 and §7 (T-00).
All code is in `infra/modal/tem_modal.py`; the nnU-Net trainer variants are in `baselines/nnunet/trainers.py`.

**Status:** the smoke check passed on 12 Nov. Results:
- Container: NVIDIA L40S (44.4 GiB usable), torch 2.15.0+cu132, CUDA available.
- Libraries: cellpose 4.2.1.1, and our nnU-Net trainers import.
- Volume: created, but the data has not been uploaded yet.

## 1. Account and budget

- **Workspace:** `muhammadfaheem-mansha`, signed in with Google as muhammadfaheem.mansha@gmail.com. The local CLI profile has the same name and is already active (`~/.modal.toml`).
- **Profile pinning:** commands below set `MODAL_PROFILE=muhammadfaheem-mansha`. The same machine also has the profiles `faheemgurkani`, `i220485` and `seecs`, so pinning stops a job going to the wrong workspace.
- **Credit:** $80. Set a workspace budget in Settings → Usage & Billing so a runaway job stops at the credit. The default spend limit is the usage limit minus credits ([Modal budgets](https://modal.com/docs/guide/budgets)).

## 2. Verified Modal facts (Context7 docs, client 1.6.4)

| Fact | Consequence for us |
|---|---|
| One Function call can run for at most **24 h** | One nnU-Net fold at 250 epochs must fit, or it must resume. `timeout=24h` |
| **GPU Functions are preemptible.** `nonpreemptible=True` is not supported for GPUs | Training must resume. `train_nnunet` retries (`modal.Retries`, up to 10) in fresh containers (`single_use_containers=True`) and passes `--c` when a checkpoint exists |
| nnU-Net writes `checkpoint_latest` only every **50 epochs** (`save_every = 50`, nnUNetTrainer.py:216) | Our trainer `nnUNetTrainer_250epochs_ckpt10` is identical to `nnUNetTrainer_250epochs` except it checkpoints every 10 epochs, so a preemption loses at most 10 epochs |
| nnU-Net only discovers trainers inside its own package | The image copies `trainers.py` into `nnunetv2/training/nnUNetTrainer/variants/tem/` |
| Volumes need `commit()` to publish writes and `reload()` to see other containers' writes; background commits also run | `_sh` commits after every command, and `train_nnunet` reloads before checking for checkpoints |
| `gpu=` accepts strings (`"L4"`, `"A10G"`, `"L40S"`, `"A100-80GB"`, `"H100"`) or a fallback list | The GPU type comes from the `TEM_MODAL_GPU` environment variable and defaults to `L40S` |
| `cpu=` counts physical cores, and Modal sets `OMP_NUM_THREADS` from it | We request 8 cores and 32 GiB of RAM. `nnUNet_n_proc_DA=8`, because nnU-Net's augmentation is CPU-bound |

## 3. Cost (approximate list prices, per hour; check modal.com/pricing)

| GPU | VRAM | GPU $/h | + 8 cores + 32 GiB (~$0.65/h) | Fits |
|---|---|---|---|---|
| L4 | 24 GB | ~0.80 | ~1.45 | ResEnc-M; ResEnc-L is borderline |
| A10G | 24 GB | ~1.10 | ~1.75 | ResEnc-M; ResEnc-L is borderline |
| **L40S (default)** | 48 GB | ~1.95 | **~2.60** | ResEnc-L with headroom |
| A100-80GB | 80 GB | ~2.50 | ~3.15 | Everything |
| H100 | 80 GB | ~3.95 | ~4.60 | Everything |

GPU prices come from third-party aggregators that agree with each other (Spheron, Beam, usagepricing.com). CPU and memory are roughly $0.047 per core-hour and $0.008 per GiB-hour.

**What $80 buys:** about 30 L40S-hours.

**Planned use, days 0–5:**

| Run | Estimate | Note |
|---|---|---|
| B1 nnU-Net fold 0, 250 epochs | $10–20 | Assumes 60–120 s per epoch, which is unmeasured (D-10). Measure it with the 5-epoch smoke trainer first (~$0.50) |
| Preprocessing (CPU only, `shell_cpu`) | under $1 | |
| B3 Cellpose fine-tune | $3–6 | |
| AxonPath inference | under $1 | Or run it locally |

That leaves at least $45 for C2/C3 runs in week 2.

## 4. Layout

| Where | What |
|---|---|
| Volume `tem-axon-myelin` → `/vol/data/raw/train-dataset/` | Challenge data. **Private workspace volume; never made public** (licence unstated, D-02) |
| `/vol/nnUNet_raw`, `/vol/nnUNet_preprocessed`, `/vol/nnUNet_results` | nnU-Net folders. The env vars are set in the image |
| `/vol/runs/` | Predictions, logs, evaluator outputs (`$TEM_RUNS`) |
| `/root/repo` (container) | The repo's code, mounted on each run. `data`, `.venv`, `external`, `docs`, `.git`, `CLAUDE.md` and all tif/zip/checkpoint files are excluded |

**Image pins:** the image is Debian slim with Python 3.12. It pins the local versions: torch 2.15.0, nnunetv2 2.8.1, cellpose 4.2.1.1 and monai 1.6.1, plus axonpath from git at 6d633bf. The image is built and cached on Modal, so editing code does not rebuild it. Editing `trainers.py` or the package list does.

## 5. Commands (from the repo root)

```bash
export MODAL_PROFILE=muhammadfaheem-mansha
M=".venv/bin/modal"

# one-time: upload the challenge data (3.1 GB) to the private volume
$M volume put tem-axon-myelin data/raw/train-dataset /data/raw/train-dataset

# smoke check: GPU, CUDA, libraries, data counts (~1 min, a few cents)
$M run infra/modal/tem_modal.py

# CPU job (conversion, plan_and_preprocess)
$M run infra/modal/tem_modal.py --action cpu --cmd "nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncL"

# measure seconds per epoch first (D-10), then the real run; --detach keeps it alive if the laptop sleeps
$M run infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_5epochs_ckpt1
$M run --detach infra/modal/tem_modal.py --action train --dataset 501 --fold 0

# results back to the local disk (never into git)
$M volume get tem-axon-myelin /nnUNet_results data/processed/modal/
$M app list; $M app logs <app-id>; $M app stop <app-id>
```

Other GPU: `TEM_MODAL_GPU=A100-80GB $M run ...`.

## 6. Pre-flight mapping (doc 12 §7 → Modal)

| §7 step | On Modal |
|---|---|
| 1. Clone repo, copy the data | The code mounts automatically. Data goes up once with `volume put` |
| 2. CUDA env, check the GPU | Done: smoke check passed on 12 Nov |
| 3. nnU-Net env vars on the large disk | Set in the image and pointing at the Volume |
| 4. T-01–T-03 | Run locally (light), then commit the split JSONs |
| 5. T-08 | `--action cpu` for conversion and preprocessing. Copy `configs/splits/dev.json` to `/vol/nnUNet_preprocessed/Dataset501_*/splits_final.json`. Then `--action train` |
| 6. Parallel work | Local evaluator work while B1 trains on Modal |
