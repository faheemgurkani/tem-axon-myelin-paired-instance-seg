# Baseline B1: nnU-Net v2 + shared watershed pairing

Method details: `docs/notes/04-methods.md` §4. Results: `docs/notes/05-results.md` §4.

| File | Role |
|---|---|
| `convert.py convert` | Challenge masks → `Dataset501_TEMAxonMyelin` (PNG, `_0000` suffix). Labels: 0 bg, 1 axon, 2 myelin, **3 ignore** (GT label 1 + orphan IDs + topology-defect fibers, which is training-label cleaning, D-21). Local CPU |
| `convert.py splits` | Writes `splits_final.json`: fold 0 = dev split, folds 1–3 = cv3. **Without it, nnU-Net makes random folds that leak overlapping tiles** |
| `trainers.py` | `nnUNetTrainer_150epochs_ckpt10_es` (used for B1), `nnUNetTrainer_250epochs_ckpt10_es`, `nnUNetTrainer_5epochs_ckpt1` (timing). Each checkpoints every 10 epochs (Modal GPUs are preemptible) and early-stops on the EMA pseudo-Dice (> 0.001 gain within 30 epochs, min 60) |

**Pipeline:**
1. `convert.py convert`, then `nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM` (local).
2. Add the `2d_bigpatch` configuration (896×1152 patch, batch 3, D-24) to the plans file. A copy is in `results/day3/nnUNetResEncUNetMPlans.json`.
3. `convert.py splits`; upload the preprocessed folder with `infra/modal/upload_dir.py`.
4. Train on Modal: `modal run --detach infra/modal/tem_modal.py --action train --trainer nnUNetTrainer_150epochs_ckpt10_es --max-gpu-hours 7`.
5. Download the `validation/*.png` files; `scripts/run/day4_postprocess.py` turns them into paired instances with `axon_myelin_seg.postprocessing.pairing`.

**Gotchas:**
- The ignore label must be the highest label and named `ignore`.
- nnU-Net only discovers trainers inside its own package, so the Modal image copies `trainers.py` there.
- nnU-Net's "New best EMA" log line is not the early-stopping rule.
