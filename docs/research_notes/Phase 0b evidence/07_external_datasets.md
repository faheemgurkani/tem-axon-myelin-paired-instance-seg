# External datasets: inventory, label mapping, overlap and the completeness audit (15 Nov 2026)

Gaps C1, C2, C5 (inventory, mapping, overlap) and the completeness audit that replaces the C4 assumption.
- **Code:**
  - `src/axon_myelin_seg/data/external.py` (`astih_to_pairs`);
  - `scripts/run/phase0b_inventory.py`, `scripts/run/phase0b_completeness_audit.py`;
  - `taxonomy.ring_candidates` gained a `px_nm` argument (the default is unchanged; tests pass).
- **Numbers:** `results/phase0b/`.
- **Overlays and galleries** (contain image pixels, so git-ignored): `data/processed/phase0b/`.
- **Local data** (git-ignored): `data/external/`.

## 1. Inventory

| | **TEM1** (DANDI 001436 v0.250512.1625) | **TEM2, DANDI** (001350 v0.250511.1527) | **TEM2, external test** (`data_axondeepseg_srf_testing` r20250513) | **AxonPath CC TEM** (Zenodo 23084690 v1) |
|---|---|---|---|---|
| Licence | CC-BY-4.0 | CC-BY-4.0 | GitHub release (same group) | CC-BY-4.0 |
| Origin | NYU (Jelescu 2016), splenium, control + cuprizone 6/12 wk | Stanford/Berkeley (Iram 2024), WT vs SRF-cKO, HPF-FS | same mice as TEM2 | AimSeg/AxonPath group; 8 of 20 images re-annotated from AimSeg (Zenodo 8351731) |
| Images (annotated) | **158**, 20 mice (6–8 each), all 3762×2286 | **10** annotated of 86, 5 mice (sub-370…375); 7 at 4675×4675 + 3 small chunks; combined `axonmyelin` mask only (axon 255, myelin 127/128) + uaxon, nuclei, process | **21** ROIs, 10 mice, 600×600 | **20** (only an annotated ROI per image), 4096×4096 float32 raw TIFFs (normalised by 1–99th percentile) |
| Pixel size | 2.36 nm | 4.93 nm | 4.93 nm | 1.75–2.11 nm raw; 8.4 nm training exports (4× down) |
| Labels | binary axon, myelin, axonmyelin PNG; ≥ 2 px gaps between fibers | + unmyelinated axon (`uaxon`), OL nucleus, OL process | axon, myelin, axonmyelin, uaxon | QuPath GeoJSON: Fibre ⊃ InnerCylinder ⊃ Axon, plus ROI; training exports as semantic + fibre-instance TIFFs |
| Split | subject-level: test = `sub-nyuMouse26` (8 images, **503 fibers**) | in-animal (see 05) | — | **tile-level** for AxonPath's model (no clean held-out images) |
| Condition per mouse | **not in metadata** (participants.tsv has species only) | in the paper (5 WT, 5 KO) | — | — |
| Fibers after mapping | **7,853** (ASTIH reports 7,711) | **4,234** (ASTIH: 4,372 myelinated) | 292 | **2,009** mapped (README 2,074; fibres without an InnerCylinder → ignore) |
| Fibers touching the border | **2,592 (33%)** | 360 (9%) | **187 (64%)** | — |
| Density (median) | 1.04 fibers/µm² (ours 0.53 per image area) | 1.04 | 1.60 | — |
| Median fiber diameter / median g | **733 nm / 0.776** | see `inventory_summary.json` | 652 nm / 0.702 | — |

**Comparison with ours:** 755 nm and g 0.750. TEM1's fibers match ours closely in size and g, so the datasets are morphologically comparable.

## 2. Label-mapping spec (C2), tested by eye
- **ASTIH → ours** (`astih_to_pairs`):
  - fiber = 8-connected component of axon ∪ myelin, valid because ASTIH enforces ≥ 2 px gaps;
  - axon part → 2k, myelin part → 2k+1;
  - components with no axon or no myelin → **1 (ignore)**, like our orphans;
  - fibers touching the border are **kept by default** (ASTIH scores them). The switch `border_to_ignore=True` mirrors the challenge convention.
- **Merged touching rings (fixed 15 Nov):**
  - Where TEM2 (and rarely TEM1) touching fibers lack the 2-px gap, a connected component holds several fibers. At first this showed up as 222 "multi-axon" components in TEM2 against 24 in TEM1.
  - **Rule added:** a component with several axon pieces is split only if its myelin encloses several axon-containing holes; each ring interior seeds a watershed (`_split_by_holes`). Fragmented axons inside one ring stay one fiber.
  - Checked by eye on `sub-370` sample 6 (`TEM2_split_check.png`): each touching fiber gets its own outline.
  - After the fix: TEM1 7,853 fibers (13 single-ring fragmented axons); TEM2 4,234 (39 fragmented).
- **Checks on TEM1:** 0 axon-only components.
  - 891 myelin-only components: **86% are specks under 0.005 µm²** (median equivalent diameter 7 nm, i.e. label noise) and 51% touch the border. Setting them to ignore is correct.
- **By-eye test** (2 images each, overlays in `data/processed/phase0b/`):
  - TEM1 `sub-nyuMouse07` samples 1–2: each axon sits inside its own ring, every fiber has its own outline, no merges.
  - TEM2 test `sub-366A` samples 1 and 4: touching fibers are separated correctly, and a myelin-only strip at the border became ignore.
  - **Passed.**
- **TEM2 extra classes:** `uaxon` (unmyelinated axon) maps to background (0), as our protocol does. OL nucleus and process also map to 0.
- **AxonPath → ours** (`axonpath_geojson_to_pairs`, done 15 Nov):
  - InnerCylinder → axon (our axon includes the tongue, D-13); Fibre − InnerCylinder → myelin;
  - each InnerCylinder goes to the Fibre it overlaps most;
  - outside the ROI, and Fibres without an InnerCylinder → ignore (1).
  - The zip's MD5 matches Zenodo (`d681abb9…`).
  - Rasterising per polygon bounding box with OpenCV `fillPoly` takes ~0.5 s per image (the first version took 25 s per image).
  - **By eye:** `e72_33_0008` and `exp69_72_EM10N_NOLPC_P5_0001` (`AxonPath_*_overlay.png`): each inner cylinder sits inside its own thin ring (remyelinating tissue). Unlabelled dark degenerating figures are left as background, as the AimSeg protocol says. **Passed.**
- **AimSeg-derived images, identified from AimSeg's Zenodo 8351731 archives:**
  - validation set: `4946_3_1_72_18_0003`, `7_EM1_3_10D_N_P_0002`, `7_EM1_3_10D_N_P_0004`, `8_E72_18_0014`, `E72_21_0003`;
  - control set: `exp69_72_EM10N_NOLPC_P5_0001`, `_0002`, `_0004`.
  - They are tagged `aimseg` in `completeness_AxonPath.csv`. The other 12 are new.

## 3. Overlap with our data (C5)
No overlap is possible. Every dataset differs from ours in microscope, pixel size and frame size:
- ours: 4096×3008 at 2.152 nm, AMT NS12;
- TEM1: 3762×2286 at 2.36 nm, Philips CM12, NYU;
- TEM2: 4.93 nm, JEOL JEM-1400, Berkeley;
- AxonPath: 1.75–2.11 nm, other lab.

No image hash check is needed. AxonPath's 8 AimSeg images are flagged by name in its `SOURCES.csv`. Don't count AimSeg as a separate dataset.

## 4. Completeness audit

**Method:**
- The same detector everywhere, at a working pixel of about 9 nm: ds 4 for ours and TEM1, ds 2 for TEM2.
- **Candidates** are unlabelled holes ≥ 300 nm enclosed by dark bands.
- **u** = candidates / (candidates + labelled fibers).
- 48 random candidates per dataset judged by eye (seed 0).
- **Rater: Claude (one non-expert rater). Provisional until a domain expert repeats it (as for T-20).**

| Dataset | Images | Labelled fibers | Candidates | Per 100 µm² of background | **u** |
|---|---|---|---|---|---|
| **Ours** (challenge train) | 266 | 8,015 | 433 | 5.06 | **5.1%** (per-image median 3.6%) |
| **TEM1** | 158 | 7,841 | 49 | 1.28 | **0.6%** |
| TEM2 test ROIs | 21 | 292 | 0 | 0 | 0, but **not interpretable** (rings cut by the 600×600 frame can't close; detector limit) |
| **TEM2 DANDI full images** | 10 | 4,234 | 60 | 2.77 | **1.4%** (per-image median 1.0%) |
| **AxonPath CC TEM** (ROI only) | 20 | 2,009 | 15 | 2.47 | **0.7%** (per-image median 0.3%) |

**By eye** (48 each):

| | Real unlabelled fiber | Interstitial gap / not a fiber | Unclear |
|---|---|---|---|
| **TEM1** | ~4 (8%), small or abnormal (one redundant-myelin figure) | **~42 (88%)**, light triangles between labelled rings | ~2 |
| **AxonPath** (all 15) | 2 (13%), both dark **degenerating** figures that the protocol excludes; **0 normal-looking** | 13 (87%) | 0 |
| **TEM2** (full images) | ~9 (19%), mostly **small thin-myelin** fibers | ~36 (75%), gaps between rings, some holding unmyelinated axons | ~3 |
| **Ours** | **~40 (83%)**: ~60% abnormal-looking (very thick or redundant "onion" myelin, large axons with thin myelin, oblique or elongated profiles), ~40% normal-looking | ~3 (6%) | ~5 |

**Reading:**
- In real unlabelled fibers per labelled fiber (u × real share): **ours ≈ 4.2% ≫ TEM2 ≈ 0.3% > AxonPath ≈ 0.1% (all degenerating) > TEM1 ≈ 0.05%.** Ours is a lower bound, because the detector misses thin myelin.
- **Overlay check:** TEM2's full images are densely annotated (62% of pixels labelled in `sub-370` sample 6). My first crop render, which coloured only one fiber, wrongly suggested sparse annotation; corrected.
- So within the detector's range (closed rings ≥ 300 nm), **TEM1's labels are near-complete**. The selective omission we found is **specific to our dataset's protocol**.
- **Caveat:** TEM1 was labelled from "full myelin rings" (Jelescu 2016). Thin or broken rings in cuprizone tissue are invisible to *both* the labeller and our detector, so TEM1 may still omit demyelinating profiles the detector can't see.

## 5. Consequences (decisions in `07-plan.md` §7.4)
1. **The C4 report's premise is reversed for TEM1, and confirmed for TEM2 and AxonPath:** all three external sets are near-complete for normal fibers; ours is the selectively labelled set. AxonPath excludes degenerating fibers by protocol, as ours does, but has no normal-looking misses. H6 must be rewritten before any model is scored. It now is (D-31, H6 pre-registered).
2. **The 13 Nov claim "fiber F1 follows label completeness"** (similar-datasets report §4.2) is **contradicted**. TEM1 has near-complete labels but the lowest published F1 (0.791). Its low precision (0.747) is therefore probably **not** selection, and our taxonomy can say what it is.
3. **C1 gains a sharper external test:** if selection errors track label incompleteness, a near-complete dataset should show a small selection share. That is H6a.
4. **The selective-omission framing (C1, C3) is supported as dataset-specific and protocol-driven.** The ~60% abnormal share of our unlabelled fibers matches the EDA (abnormal fibers excluded on purpose) and the random-miss estimate.
5. **TEM1's per-mouse condition is not in the metadata**, so "score control animals only" can't be done as planned.
   - Proxy: stratify mice by GT median g (cuprizone raises g); report it as a proxy.
   - Or ask the ASTIH authors for the mapping.
