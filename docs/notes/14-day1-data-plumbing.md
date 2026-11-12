# 14 — Day 1: data plumbing, label cleaning, splits (T-01, T-02, T-03), executed 13 Nov 2026

Part of the day 1–5 execution of [09](09-research-plan.md). Each section follows the same order:
- what was done, with code paths;
- the numbers, from committed files under `results/`;
- how they compare with the plan's expectation;
- the things that are easy to miss.

All numbers here are reproducible with `python scripts/run/day1_data.py` (≈ 22 s on the M4, 4 workers).

## Environment incident before starting (fixed)

**Symptom:** `from axon_myelin_seg.data.cleaning import …` failed with *ModuleNotFoundError*, although the file existed.

**Cause:**
- The repo moved from `~/Local Projects/research/…` to the external disk (`/Volumes/Sandisk 2TB/…`) on 12 Nov.
- The `.venv` editable installs (`axon-myelin-seg` and `axonpath`) still pointed at the **old copy**, which still exists on disk.
- Python therefore imported stale code from the old location without any error.

**Fix:**
- `uv pip install --no-deps -e .` and `uv pip install --no-deps -e external/axonpath` from the new location.
- Verified: `axon_myelin_seg.__file__` and `axonpath.__file__` now resolve under `/Volumes/Sandisk 2TB/…`.

**Lesson:**
- After moving a repo, check `python -c "import pkg; print(pkg.__file__)"` for every editable package.
- A stale editable install fails *silently* when the old copy still exists: you run old code and get no error.
- The old copy at `~/Local Projects/…/tem-axon-myelin-paired-instance-seg` was left untouched. It is the owner's to delete.

## T-01 Loader — `src/axon_myelin_seg/data/io.py`

**What it does:**
- `list_samples()` returns the 266 names that have both an image and a mask.
- `load_sample(name)` returns (uint8 image, uint16 mask). It validates the shape (3008, 4096) and the dtypes and raises an error on any mismatch.

**Result:** 266/266 load, 0 shape or dtype mismatches (`results/day1/images.csv`, column `ok`). **Matches the expectation.**

**Easy to miss:**
- Shapes are **(H, W) = (3008, 4096)** in numpy, but the challenge text says "4096×3008" (W×H). Mixing them up silently transposes crops.
- Masks are uint16 by the challenge convention. The largest GT ID in the training set is only 189 (`data/processed/eda/images.csv`, `max_id`), so uint8 would hold the GT today. But a *prediction* with more than 127 fibers needs IDs above 255. Always write uint16; casting a prediction to uint8 wraps IDs and silently breaks the X/X+1 pairing.

## T-02 Training-label cleaner — `src/axon_myelin_seg/data/cleaning.py`

**What it does:** returns a copy of the mask in which these become label 1 (ignore):
- **Orphan IDs:** an axon X without its X+1, or a myelin X+1 without its X.
- **Topology-defect fibers:** any of the following holds:
  - the myelin has more than one 8-connected component;
  - the myelin's Euler number is not 0, i.e. it is not exactly one ring;
  - the axon is not entirely inside the filled myelin.

**Result** (`results/day1/cleaning_summary.json`):

| Quantity | Plan expectation (EDA) | Measured |
|---|---|---|
| Orphan IDs | 47 | **47** |
| Topology-defect fibers | ~296 | **296**, in 116 images |
| Extra ignored pixels | — | 42.6 M px, 1.3% of all pixels |
| Paired fibers in total | 7,968 | **7,968** |

**Matches the expectation exactly**: the definition is the one in `scripts/eda/fiber_stats.py`.

**Easy to miss:**
- **The cleaner is for training labels only.** The evaluation GT is never cleaned, because the challenge scores the raw GT. Cleaning the eval GT would inflate our scores and break comparability.
- "Defect" means *annotation geometry that contradicts the encoding*, for example a ring drawn as a "C" (Euler 1) or an axon spilling past its ring. The label is not necessarily wrong about the biology. We ignore these fibers rather than delete them: ignored pixels give no loss, so the model is neither told "this is background" nor taught the broken shape.
- Ignoring costs supervision on 3.7% of fibers. Accepted (D-21).

## T-03 Splits — `src/axon_myelin_seg/data/splits.py`

**What it does:**
1. Groups:
   - **v2:** 52 images form 22 overlap groups (same tissue, `configs/splits/overlap_groups.json`);
   - every other image is its own group;
   - total **236 groups** = 266 − 52 + 22;
   - v1 (EDA) had 40 images in 18 groups, so 244 groups; it is kept as `overlap_groups_v1_eda.json`.
2. Stratum per image: the fiber-density tercile of its group's mean density. Density = paired fibers per µm² of non-ignored area.
3. `StratifiedGroupKFold(shuffle=True, random_state=0)`:
   - 5 folds, of which fold 0 is the **dev split** (`configs/splits/dev.json`);
   - 3 folds give `configs/splits/cv3.json`.
4. `check_split` asserts three things: train and val are disjoint, together they cover all images, and **no group crosses the split**. Tests: `tests/test_day1_data.py`.

**Result: v2, the frozen version** (regenerated on day 2 after T-07 found 8 more overlapping pairs; see "Revision" below):

| Split | Train / val images | Val fibers | Val median density | Train median density |
|---|---|---|---|---|
| dev (fold 0 of 5) | **213 / 53** | 1,619 | 0.605 /µm² | 0.601 /µm² |
| cv3 fold 0 | 177 / 89 | 2,650 | 0.605 | 0.592 |
| cv3 fold 1 | 177 / 89 | 2,711 | 0.592 | 0.605 |
| cv3 fold 2 | 178 / 88 | 2,607 | 0.604 | 0.604 |

Density ranges from 0.108 to 1.471 fibers/µm² (13.7×, consistent with E9).

**Matches the expectation:** ~213/53, and no group crosses a split. The stratification works: dev val and train medians differ by 0.001/µm².

**Easy to miss:**
- **Never split by tile or random image** in this dataset. Overlapping tiles are the *same axons* annotated twice. Putting one in train and one in val makes the validation score measure memorisation.
- The stratum is assigned per **group**, not per image. Otherwise StratifiedGroupKFold receives conflicting labels inside a group.
- Both JSONs are frozen and committed, with the method string and seed inside. Never regenerate them after results exist: every number from day 3 on is tied to these exact lists.
- nnU-Net receives these splits through `splits_final.json` (T-08): fold 0 = dev, folds 1–3 = cv3. **nnU-Net's own `fold 0` would otherwise be a random 5-fold split that ignores groups.**

## Revision on day 2: the v1 split leaked, so the splits were regenerated before any training

**What happened:**
- T-07 found a registration bug in the EDA (doc [15](15-day2-evaluator.md) §T-07).
- With it fixed, 8 more tile pairs turned out to show the same tissue: 0113/0233, 0045/0252, 0123/0181, 0018/0238, 0148/0218, 0104/0263, 0223/0238 and 0139/0224.
- Evidence: image NCC ≥ 0.82 after correct registration. In 4 pairs the annotations also agree at axon Dice ≥ 0.98. The other 4 have almost no labelled fibers in the overlap, so they were checked visually; side-by-side crops in `data/processed/day2/new_overlap_check.png` (local) show identical tissue.
- **Five of the 8 crossed the v1 dev split** (one tile in train, the other in val): 0113/0233, 0045/0252, 0018/0238, 0223/0238 and 0139/0224.

**What was done:**
- The pairs were merged into the groups by union-find. 0238 joins the 0018 and 0223 groups into one.
- Splits regenerated with the same code and seed. 39 of the 53 dev-val images changed.
- **No model had been trained or evaluated on v1**, so no result is affected. The v1 files are kept for traceability:
  - `results/day1/dev_v1_superseded.json`
  - `results/day1/cv3_v1_superseded.json`
  - `configs/splits/overlap_groups_v1_eda.json`

**Remaining risk:**
- The overlap search only covered the 710 candidate pairs in the EDA's ORB scan, so other overlaps may exist.
- Any overlap found later must be checked against the frozen splits. If one crosses the split, report it as a limitation; don't silently re-split after results exist.

## Status

- [x] T-01
- [x] T-02
- [x] T-03

Tests: `pytest tests/test_day1_data.py` (5 tests, data-dependent ones skip without data).
