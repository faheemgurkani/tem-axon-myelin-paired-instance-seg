# 13 — Cloud compute on Modal (12–13 Nov 2026)

Modal replaces the dedicated GPU machine planned in [12](12-day0-5-resources.md) §1 and §7 (T-00).

**Where things are:**
- Modal app: `infra/modal/tem_modal.py`
- nnU-Net trainer variants: `baselines/nnunet/trainers.py`
- Their tests: `tests/test_nnunet_early_stopping.py`

**Rule (owner, 13 Nov):** CPU work runs on the local M4. Modal runs only GPU work, and only for as long as the GPU is needed (§3b).

**Status:**
- **12 Nov, L40S smoke check:** passed with torch 2.15.0+cu132 and 44.4 GiB.
- **12 Nov, data:** uploaded to the Volume (266 images + 266 masks, verified).
- **13 Nov, L4 smoke check:** passed with 22.1 GiB. The early-stopping trainer imports, and the Volume is reachable.
- **No training has run.**

## 1. Account and budget

- **Workspace:** `muhammadfaheem-mansha`, signed in with Google as muhammadfaheem.mansha@gmail.com. The local CLI profile has the same name and is already active (`~/.modal.toml`).
- **Profile pinning:** commands below set `MODAL_PROFILE=muhammadfaheem-mansha`. The same machine also has the profiles `faheemgurkani`, `i220485` and `seecs`, so pinning stops a job going to the wrong workspace.
- **Credit:** $80 in the account. **Budget for days 0–5: $10** (owner, 12 Nov).
- **Spend cap:** set a workspace budget in Settings → Usage & Billing so a runaway job stops. The default spend limit is the usage limit minus credits ([Modal budgets](https://modal.com/docs/guide/budgets)).

## 2. Verified facts (Modal docs via Context7, client 1.6.4; nnU-Net 2.8.1 as installed)

| Fact | Consequence for us |
|---|---|
| One Function call can run for at most **24 h** | `timeout=24h`. Our GPU-hour cap (§3b) is far below this |
| **GPU Functions are preemptible.** `nonpreemptible=True` is not supported for GPUs | Training must resume. `train_nnunet` retries (up to 3) in fresh containers (`single_use_containers=True`) and passes `--c` when any checkpoint exists |
| nnU-Net 2.8.1 writes `checkpoint_latest.pth` only every **50 epochs** (`save_every`) | Our trainers checkpoint every 10 epochs, so a preemption loses at most 10 epochs |
| nnU-Net 2.8.1 (pip, used in the image) writes `.pth` checkpoints. The newer GitHub master in `external/nnUNet` writes `.safetensors` | The resume check matches `checkpoint_*` of either format, so a later version bump won't silently restart training from scratch |
| nnU-Net 2.8.1 stores preprocessed data as blosc2 `.b2nd`, read in place with no unpacking step | The GPU container copies the preprocessed dataset to its local disk in seconds; no CPU-heavy unpack happens on Modal |
| nnU-Net already tracks an EMA of the validation pseudo-Dice (`ema_fg_dice`) to choose `checkpoint_best` | Early stopping uses the same signal; no extra GPU work is needed to compute it |
| `nnUNetv2_train --val_best` runs the final validation with `checkpoint_best` | Early-stopped runs validate with the best weights |
| nnU-Net only discovers trainers inside its own package | The image copies `trainers.py` into `nnunetv2/training/nnUNetTrainer/variants/tem/` |
| Volumes need `commit()` to publish writes and `reload()` to see other containers' writes | Every function reloads at the start and commits at the end |
| `cpu=` counts physical cores, and Modal sets `OMP_NUM_THREADS` from it | 6 cores, 16 GiB of RAM, and `nnUNet_n_proc_DA=6` (§3b) |

## 3. Cost (approximate list prices, per hour; check modal.com/pricing)

| GPU | VRAM | GPU $/h | + 6 cores + 16 GiB (~$0.41/h) | Fits |
|---|---|---|---|---|
| **L4 (default)** | 24 GB | ~0.80 | **~1.21** | ResEnc-M easily (8 GB target); ResEnc-L (24 GB target) is borderline |
| A10G | 24 GB | ~1.10 | ~1.51 | ResEnc-M; ResEnc-L is borderline |
| L40S | 48 GB | ~1.95 | ~2.36 | ResEnc-L with headroom |
| A100-80GB | 80 GB | ~2.50 | ~2.91 | Everything |
| H100 | 80 GB | ~3.95 | ~4.36 | Everything |

GPU prices come from third-party aggregators that agree with each other (Spheron, Beam, usagepricing.com). CPU and memory are roughly $0.047 per core-hour and $0.008 per GiB-hour.

## 3a. GPU choice for a $10 budget (12 Nov)

**Choice: L4 + nnU-Net ResEnc-M** (replaces L40S + ResEnc-L).

**Why a big GPU isn't needed:**
- This is a 2D U-Net with tens of millions of parameters, not a billion-parameter LLM.
- ResEnc-M is planned for 8 GB of VRAM and ResEnc-L for 24 GB (nnU-Net planners).
- At this scale nnU-Net 2D is usually limited by CPU augmentation, not GPU compute.

**Why L4:**
- It is the cheapest Modal GPU that fits ResEnc-M with room to spare.
- Its fp16 tensor throughput is comparable to an A10G.
- Its weakness is memory bandwidth (about 300 vs 600 GB/s for A10G). So it may be up to ~1.5× slower per epoch, at ~0.7× the price.
- A T4 is cheaper per hour but roughly 2× slower.

**Why ResEnc-M:**
- The plan allowed it already (D-09/D-10).
- ResEnc-L costs roughly 2× the compute per step, so a full L run likely doesn't fit in $10.

**What $10 buys:** about 8 L4-hours at ~$1.21/h.

## 3b. Compute split and GPU-seconds rules (13 Nov)

**Where each step runs:**

| Step | Where | Why |
|---|---|---|
| Convert the data to nnU-Net format (labels {0, 1 axon, 2 myelin, 3 ignore}) | **Local M4** | Pure CPU and file I/O |
| `nnUNetv2_plan_and_preprocess` | **Local M4** (`-np 4`; ~16 GB RAM is enough for 2D) | CPU only. Its output (`.b2nd`) is uploaded once with `modal volume put` |
| Write `splits_final.json` from `configs/splits/dev.json` | **Local**, then uploaded with the preprocessed folder | Trivial |
| Training (forward/backward passes plus on-the-fly augmentation) | **Modal GPU** | Augmentation **cannot** move to the Mac. Every batch would cross the network, and the GPU would idle waiting. The GPU container's own cores do it, which is why it requests 6 cores |
| Final validation (sliding window, softmax `.npz`) | **Modal GPU**, inside the training call | It needs the GPU and the weights are already loaded |
| Inference on new images (`--action predict`) | **Modal GPU** | Inputs are prepared locally and uploaded |
| Pairing, evaluation, oracle analysis, figures | **Local M4** | CPU only. Download `.npz` and predictions with `modal volume get` |
| AxonPath zero-shot (T-11) | **Local M4** (MPS, ~2 s per image) | Already measured: 266 images take ~10 min locally, so a cloud GPU isn't worth paying for |

**How GPU seconds are kept to the minimum:**
1. **Early stopping** (`nnUNetTrainer_250epochs_ckpt10_es`, the default):
   - The trigger is the EMA pseudo-Dice failing to improve by more than 0.001 for 30 epochs.
   - It never stops before epoch 60, and 250 epochs remains the cap.
   - Expect it to stop somewhere between epochs 100 and 200, which saves an estimated 20–60% of the GPU time. This is an estimate; log the actual stop epoch.
   - The rule reads the logged history, so it still holds after a preemption resume.
   - Final validation uses `--val_best`.
   - **Caveat:** nnU-Net's poly learning-rate schedule is defined over 250 epochs, so an early-stopped run ends at a higher learning rate than a full run. Say this in the write-up.
2. **Hard GPU-hour cap per run** (`--max-gpu-hours`, default 5):
   - A ledger on the Volume (`/vol/runs/budget/<run>.json`) fixes the deadline at the first attempt, so retries after preemption can't extend it.
   - At the deadline the job is stopped, and `checkpoint_latest` is at most 10 epochs old.
   - A later call with the same run key refuses to start once its cap is spent. To continue on purpose, delete or raise the ledger file.
3. **No idle containers:** each call runs in a container that exists only for that call (`single_use_containers=True`). No warm pool and no web endpoints, so billing ends when the function returns.
4. **Local-disk staging:** the preprocessed `.b2nd` files are copied to container-local disk before training, so patch reads never stall on network storage. That costs seconds of GPU time.
5. **GPU-utilisation log:**
   - `nvidia-smi` is sampled every 15 s into `/vol/runs/gpu_util/<run>.csv`, and a summary is printed at the end.
   - If mean utilisation is below ~70%, the CPU workers are starving the GPU and paid GPU seconds are idle. Then raise `CPU` (cores cost ~$0.05/h, much less than idle L4 time) or reduce augmentation cost.
6. **Measure before committing:** a 5-epoch timing run (`nnUNetTrainer_5epochs_ckpt1`, `--max-gpu-hours 0.3`, ≈ $0.35) gives the seconds per epoch and the GPU utilisation before the long run is launched.
   - The first epoch includes `torch.compile` warm-up, so use epochs 2–5 for the timing.

**Spending plan ($10):**

| Item | Estimate |
|---|---|
| Smoke checks (done) | ~$0.05 |
| 5-epoch timing run | ~$0.35 |
| B1 fold 0 with early stopping | ~$3–6 if an epoch takes 50–80 s and the run stops at epochs 100–200 |
| Spare for a re-run or Cellpose (T-10) | the rest |

**Rule:** if the timing run implies a 250-epoch cap would cost more than $8, lower the cap (`--max-gpu-hours`) and log it under D-10.

## 4. Layout

| Where | What |
|---|---|
| Volume `tem-axon-myelin` → `/vol/data/raw/train-dataset/` | Challenge data. **Private workspace volume; never made public** (licence unstated, D-02) |
| `/vol/nnUNet_preprocessed/Dataset<ID>_<name>/` | Preprocessed on the Mac and uploaded, including `splits_final.json` |
| `/vol/nnUNet_results/` | Checkpoints, `progress.png`, training logs, validation `.npz` |
| `/vol/runs/{budget,gpu_util}/` | GPU-hour ledgers and utilisation logs |
| `/root/repo` (container) | The repo's code, mounted on each run. `data`, `.venv`, `external`, `docs`, `.git`, `CLAUDE.md` and all tif/zip/checkpoint files are excluded |

**Image pins:** the image is Debian slim with Python 3.12. It pins the local versions: torch 2.15.0, nnunetv2 2.8.1, cellpose 4.2.1.1 and monai 1.6.1, plus axonpath from git at 6d633bf. Code edits don't rebuild the image; editing `trainers.py` or the package list does.

## 5. Commands (from the repo root)

```bash
export MODAL_PROFILE=muhammadfaheem-mansha
M=".venv/bin/modal"
export nnUNet_raw="$PWD/data/nnunet/raw" nnUNet_preprocessed="$PWD/data/nnunet/preprocessed" nnUNet_results="$PWD/data/nnunet/results"

# --- local (CPU, M4) ---
# T-08 conversion (script to be written in T-08) -> $nnUNet_raw/Dataset501_TEMAxonMyelin
.venv/bin/nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM -np 4
cp configs/splits/dev_nnunet.json "$nnUNet_preprocessed/Dataset501_TEMAxonMyelin/splits_final.json"
$M volume put tem-axon-myelin "$nnUNet_preprocessed/Dataset501_TEMAxonMyelin" /nnUNet_preprocessed/Dataset501_TEMAxonMyelin

# --- Modal (GPU only) ---
$M run infra/modal/tem_modal.py                                                  # smoke check (~1 min)
$M run infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_5epochs_ckpt1 --max-gpu-hours 0.3
$M run --detach infra/modal/tem_modal.py --action train --dataset 501 --fold 0 --max-gpu-hours 5

# --- back to local for analysis (never into git) ---
$M volume get tem-axon-myelin /nnUNet_results data/nnunet/results_modal/
$M volume get tem-axon-myelin /runs data/processed/modal_runs/
$M app list; $M app logs <app-id>; $M app stop <app-id>
```

Other GPU: `TEM_MODAL_GPU=A10G $M run ...`.

## 6. Pre-flight mapping (doc 12 §7 → Modal)

| §7 step | Where | Status |
|---|---|---|
| 1. Code and data on the GPU side | The code mounts automatically; the data is on the Volume | Done 12 Nov |
| 2. CUDA env, check the GPU | Modal | Done (L40S 12 Nov, L4 13 Nov) |
| 3. nnU-Net env vars | Set in the image (Volume) and locally (`data/nnunet/`) | Done |
| 4. T-01–T-03 (loader, cleaner, splits) | Local | Next |
| 5. T-08 conversion and preprocessing | Local, then `volume put` | After T-03 |
| 5b. T-08 timing run, then the training run | Modal, `--action train` | After 5 |
| 6. Evaluator (T-04–T-07) and AxonPath (T-11) | Local, in parallel with B1 training | — |
