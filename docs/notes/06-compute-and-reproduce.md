# 06 — Compute, environment, budget and how to reproduce everything

## 1. Machines and the compute split

**Local machine:** Apple M4, 10 cores, 16 GB RAM, MPS GPU (no CUDA). The repo sits on the 2 TB external disk.

**Owner's rule (13 Nov):**
- **All CPU work runs locally:** conversion, nnU-Net planning and preprocessing, pairing, evaluation, AxonPath and Cellpose zero-shot (on MPS).
- **Modal runs GPU work only**, and only while the GPU is needed.

nnU-Net's on-the-fly augmentation must run inside the GPU container (6 cores), because shipping batches over the network would idle the GPU.

## 2. Python environment

- `.venv/` (git-ignored): uv, Python 3.12.
- Versions: torch 2.15.0, nnunetv2 2.8.1, cellpose 4.2.1.1, monai 1.6.1, scikit-image 0.26, numpy 2.5.4.
- Editable installs: the project and `external/axonpath` (@ 6d633bf).
- Third-party repos in `external/` (git-ignored, shallow clones):
  - nnUNet @ 7bb9261 (master writes `.safetensors`; the pip 2.8.1 we run writes `.pth`);
  - cellpose @ a54cb48;
  - axonpath @ 6d633bf;
  - ASTIH @ 089896e;
  - axondeepseg @ 05a8dd1.
- `stardist` is not installed (it pulls TensorFlow). Its matching is reimplemented in the evaluator.
- Tests: `.venv/bin/pytest -q tests` (20 tests). Bare `pytest` also collects `external/nnUNet`'s tests, which error.

**Repo-move incidents (both fixed; a venv is not relocatable):**
1. **Day 1:** the editable installs still pointed at the old copy in `~/Local Projects/…`, so code was silently imported from there.
   - Fix: `uv pip install --no-deps -e .` and `uv pip install --no-deps -e external/axonpath`.
2. **Day 4:** 109 launchers in `.venv/bin` (`activate`, `pytest`, `modal`, `nnUNetv2_*`) still had the old venv path in their shebang. `source .venv/bin/activate` therefore ran the **old** venv.
   - Fix: rewrote the path in every file.
   - **Earlier results are unaffected:** they were run with `.venv/bin/python`, and the old copy only has stub modules, so a wrong run crashes rather than giving wrong numbers.

**Check after any move:**
- `source .venv/bin/activate; which python`
- `python -c "import axon_myelin_seg; print(axon_myelin_seg.__file__)"`

The old copy in `~/Local Projects/` is left for the owner to delete.

## 3. Modal

**Account and profile:**
- Workspace and CLI profile `muhammadfaheem-mansha`; $80 of credit.
- **Always prefix commands with `MODAL_PROFILE=muhammadfaheem-mansha`.** Other profiles on this machine (faheemgurkani, i220485, seecs) must not be billed.

**App:** `infra/modal/tem_modal.py`, with actions `check` (GPU smoke test), `train`, `predict` and `gpu`.
- **Image:** Debian slim, Python 3.12, pinned to the local versions. The trainers file is copied into nnU-Net's package. Code is mounted at `/root/repo` (data, `.venv`, `external`, `docs` excluded).
- **Default resources:** L4 GPU (24 GB), 6 cores, 16 GiB RAM, ≈ $1.21/h. Override with `TEM_MODAL_GPU=A10G|L40S`.

**Why L4 + ResEnc-M:**
- This is a 2D U-Net with tens of millions of parameters, not an LLM.
- ResEnc-M targets 8 GB of VRAM, and L4 is the cheapest GPU that fits it comfortably.
- The timing run showed 97% GPU utilisation, so it is compute-bound and more cores wouldn't help.

**Volume `tem-axon-myelin`** (private, **never make it public**; 21.4 GB on 13 Nov):

| Path | Content | Size |
|---|---|---|
| `/data/raw/train-dataset/` | 266 images + 266 masks | 3.3 GB |
| `/nnUNet_preprocessed/Dataset501_TEMAxonMyelin/` | Preprocessed locally and uploaded, plus `splits_final.json` | 7.7 GB |
| `/nnUNet_results/` | Checkpoints, logs, validation PNG / npz / pkl | 10.4 GB (6.8 GB is B1's npz) |
| `/runs/budget/`, `/runs/gpu_util/` | GPU-hour ledgers, nvidia-smi logs | — |

**Mechanisms that avoid paying for idle GPU seconds:**

| Mechanism | How |
|---|---|
| Early stopping | See [04](04-methods.md) §4 |
| Hard GPU-hour cap per run (`--max-gpu-hours`) | A ledger on the Volume fixes the deadline at the first attempt, so preemption retries can't extend it. A later call refuses once the cap is spent |
| Preemption safety | GPU functions are preemptible. Checkpoints every 10 epochs; up to 3 retries, which resume with `--c` when any `checkpoint_*` (`.pth` or `.safetensors`) exists |
| No idle containers | `single_use_containers=True`, no warm pool |
| Fast staging | Preprocessed data is copied to the container disk with 32 threads: **3 s** (sequential `copytree` took 376 s, wasting 6 GPU-minutes per start) |
| Utilisation log | nvidia-smi every 15 s. Below ~70% means the CPU is starving the GPU |
| Measure first | A 5-epoch timing run (≈ $0.37) before any long run |

**Volume semantics:** `commit()` publishes writes and `reload()` sees other containers' writes. The 24 h function limit is far above our caps.

**Other gotchas met:**
- **A silent upload failure.** `modal volume put` of the 7.1 GB folder failed with a connection error, but piped through `tail` it returned exit 0.
  - Fix: `infra/modal/upload_dir.py` uploads 40 files per batch with 5 retries and skips files already present with the same size.
  - **Never trust a piped exit code**: use `set -o pipefail` and count files at the destination.
- **A path bug.** `parents[2]` failed inside the container. Fix: `REPO = parents[2] if modal.is_local() else /root/repo`.
- **Downloading a directory** with `modal volume get` would also pull the 6.8 GB of npz. Fetch files individually, in parallel, and count them; 2 of 53 needed a retry.

## 4. Budget (days 0–5 cap: $10)

| Item | Cost (Modal billing) |
|---|---|
| Smoke checks (L40S 12 Nov, L4 13 Nov) | $0.02 |
| Timing run (18 min) | $0.37 |
| **B1** (6.34 L4-hours) | **$7.69** |
| **Total compute, Nov 2026 (`modal billing summary`)** | **$8.16**, covered by credit |
| Volume storage | $0, within the free allowance (metered $6.81) |

**$1.84 is left.** Any further GPU work needs the owner to raise the cap first.

Check spend with `modal billing summary` / `modal billing report --for "this month"`.

## 5. Reproduce (from the repo root)

```bash
P=.venv/bin/python; M=.venv/bin/modal; export MODAL_PROFILE=muhammadfaheem-mansha
export nnUNet_raw=$PWD/data/nnunet/raw nnUNet_preprocessed=$PWD/data/nnunet/preprocessed nnUNet_results=$PWD/data/nnunet/results

# Day 1 (local, ~22 s): loader check, cleaning counts, splits → results/day1/
$P scripts/run/day1_data.py

# Day 2 (local): human ceilings → results/day2/
$P scripts/run/day2_ceilings.py && $P scripts/run/day2_ceilings_pool.py

# Day 3 (local): R1 (~15 min), nnU-Net data, AxonPath, Cellpose zero-shot
$P scripts/run/day3_r1_pairing_on_gt.py
$P baselines/nnunet/convert.py convert && $P baselines/nnunet/convert.py splits
.venv/bin/nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM -np 4
#   then add "2d_bigpatch" to the plans file (copy in results/day3/nnUNetResEncUNetMPlans.json)
MODAL_PROFILE=muhammadfaheem-mansha $P infra/modal/upload_dir.py "$nnUNet_preprocessed/Dataset501_TEMAxonMyelin" /nnUNet_preprocessed/Dataset501_TEMAxonMyelin
$P baselines/axonpath/run.py --split val
$P baselines/cellpose_sam/zeroshot.py

# Day 3–4 (Modal GPU): timing run, then B1
$M run infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_5epochs_ckpt1 --max-gpu-hours 0.3
$M run --detach infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_150epochs_ckpt10_es --max-gpu-hours 7
#   download fold_0/validation/*.png + summary.json into data/nnunet/results_modal/... (files one by one; skip the npz)

# Day 4–5 (local, ~25 min)
V=data/nnunet/results_modal/nnUNet_results/Dataset501_TEMAxonMyelin/nnUNetTrainer_150epochs_ckpt10_es__nnUNetResEncUNetMPlans__2d_bigpatch/fold_0/validation
$P scripts/run/day4_postprocess.py --val-dir $V
for n in axonpath b1 cellpose_b1; do $P scripts/run/score_predictions.py --pred-dir data/predictions/$n --name $n; done
$P scripts/run/day4_oracles.py --pred-dir data/predictions/axonpath --name axonpath
for n in b1 cellpose_b1; do $P scripts/run/day4_oracles.py --pred-dir data/predictions/$n --name $n --sem-dir data/predictions/b1_sem; done
$P scripts/run/day5_analysis.py
$P scripts/run/day4_fp_gallery.py --pred-dir data/predictions/b1 --cls other_bg --n 16 --out data/processed/day4/b1_other_bg_gallery.png
```

## 6. Where files live

| Path | Content | In git? |
|---|---|---|
| `configs/splits/` | Overlap groups (v2 + v1), dev and cv3 splits | Yes (frozen) |
| `results/day1..day5/` | Metric tables, JSON summaries, run logs | Yes (no images or masks) |
| `data/raw/` | Challenge data | **Never** |
| `data/nnunet/{raw,preprocessed,results_modal,logs}` | nnU-Net folders, B1 outputs and logs | No |
| `data/predictions/<baseline>/` | uint16 prediction maps | No |
| `data/processed/` | EDA tables, galleries, GPU-utilisation CSVs | No |
| `results/day5/fibers_*.csv` | Per-fiber GT-derived table | No (licence caution) |
