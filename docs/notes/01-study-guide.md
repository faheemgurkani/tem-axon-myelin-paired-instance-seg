# 01 — Study guide: the whole project, start to now (state on 13 Nov 2026)

Read this file first. It tells the story once and gives only the headline numbers. Each section points to the file holding the details.

| File | What it holds |
|---|---|
| **01-study-guide.md** | This file: problem, gap, hypotheses, what we did, what we found, where we stand |
| [02-challenge-and-data.md](02-challenge-and-data.md) | The challenge rules, and everything we know about the data (EDA) |
| [03-literature-and-gap.md](03-literature-and-gap.md) | Prior work and the research gap, as validated |
| [04-methods.md](04-methods.md) | How every number is produced: splits, evaluator, oracles, baselines |
| [05-results.md](05-results.md) | All results from days 1–5, with tables and caveats |
| [06-compute-and-reproduce.md](06-compute-and-reproduce.md) | Environment, Modal, budget, incidents, commands to reproduce |
| [07-plan.md](07-plan.md) | Hypotheses, decision register (D-xx), next tasks, risks, log |

Longer source material, kept unchanged:
- `docs/reports/` holds the two deep-research reports (11 and 12 Nov).
- `docs/research_notes/` holds their source notes.
- The old day-by-day notes (00–18) are in git history before the consolidation commit.

---

## 1. The problem in plain words

**The images:**
- Transmission electron microscopy (TEM) of mouse brain white matter (corpus callosum), cut across.
- Each **myelinated fiber** looks like a light disc (the **axon**, the "wire") inside a dark ring (the **myelin** sheath, the "insulation").
- 266 annotated images of 4096×3008 px at 2.152 nm/px, with about 30 fibers each.

**The output:** for every fiber, its axon gets an even ID `X` and *its own* ring gets `X+1`. This is **paired instance segmentation**: instances that each have two nested parts.

**Why the pairing matters:** the biological readout is the **g-ratio** = axon diameter ÷ fiber diameter.
- It is about 0.75 in healthy mouse brain.
- It rises when myelin thins (disease) and falls back when myelin repairs.
- It can only be computed per fiber if each axon is linked to its own ring.

**How the challenge scores it** (SIPAIM 2026; we are not registered and use it only as a problem statement):
- **Dice** for axon pixels and myelin pixels.
- **HD95**, a boundary distance in nm.
- **g-MAE**, the g-ratio error over *matched* fibers.
- These are combined by rank.
- **Trap:** g-MAE ignores fibers the model missed, so a model can look better by predicting fewer, easy fibers. We always report fiber recall next to it.

Details: [02](02-challenge-and-data.md) §1.

## 2. The gap and the original bet

**What the literature shows** ([03](03-literature-and-gap.md)):
- Every published axon/myelin tool first labels pixels (axon / myelin / background).
- It then *pairs* axons with rings using a hand-written rule, usually a watershed.
- No tool makes "one axon inside one ring" a property of the network's output. None trains on g-ratio error, and none measures whether pairing is correct.

**The headline motivation:** ASTIH (MIDL 2026) reports pixel Dice 0.97 / 0.92, but only 0.79 fiber detection F1 after watershed grouping.
- Read naively, this says "pairing is the bottleneck".
- Our gap check (12 Nov) found that this F1 measures *detection*, not pairing.
- Also, AxonPath (Sep 2026) already predicts fiber and axon cues jointly; it still pairs by a rule.
- So the gap is real but narrow. The defensible novelty is a dense decoder in which pairing holds *by construction*, plus a metric that scores pairing.

**Original bet, contribution C2:** a "nested-by-construction" decoder.
- It predicts whole fibers first, then splits each fiber into axon and ring.
- A ring can then never belong to the wrong axon.

**Why we did not build it straight away:** the bet assumes pairing errors are a large part of fiber-level error. Our EDA suggested a rival cause: **selection**. Annotators skip some real fibers and deliberately leave abnormal fibers as background. So plan v2 (12 Nov) became **evaluation-first**:
1. Build an evaluator that attributes each error to one cause: selection, separation/pairing or boundary (contribution **C1**, always delivered).
2. Train a strong standard baseline (**B1** = nnU-Net + watershed).
3. Use "oracles" (§5) to measure how much each cause costs.
4. Decide on day 5 with a pre-written rule (**H1**) whether C2 is worth building.
   - If not, switch to **C3**: training that tolerates incompletely annotated fibers.

**H1 (GO if all three hold):**
- fixing separation with an oracle gains ≥ 0.03 fiber F1;
- that gain is at least as large as the gain from fixing selection;
- separation errors are ≥ 30% of all fiber errors.

## 3. What the data taught us before any model ([02](02-challenge-and-data.md) §2–3)

1. **The GT pairing is clean.** 1 axon per ring, 99.5% of axons fully enclosed, no axon touches another fiber. A nested output format matches the labels exactly.
2. **71% of fibers touch a neighbour.** If pairing is hard anywhere, it is here. The light triangle between three touching rings looks like a fiber: an "interstitial gap" trap.
3. **Selection is a hidden task.**
   - ~4–8% of real fibers are left unlabelled at random.
   - ~25% of ring-like structures sit in background, mostly abnormal fibers (thin, split or degenerating myelin) that the protocol leaves out.
   - Background is therefore not clean "negatives".
4. **Label 1 ("ignore") is mostly fibers, not debris.** It marks border-cut fibers and some deliberately excluded large or oblique fibers, covering 15.8% of pixels. Predictions there are neither right nor wrong.
5. **Overlapping tiles exist.** 52 images form 22 groups showing the same tissue. Splitting them carelessly leaks validation. On the plus side, they give a free "second annotator" to measure the human ceiling.
6. **g is sensitive:** 1 px of boundary shift ≈ 0.006 in g, which equals the human noise floor. Different g formulas differ by 0.065, so the exact formula matters more than the method.

## 4. What we did, day by day (13 Nov)

| Day | Work | Outcome in one line |
|---|---|---|
| 0 | Resources, environment, Modal GPU account ($10 cap for days 0–5) | L4 GPU + nnU-Net ResEnc-M chosen; CPU work runs locally |
| 1 | Loader, training-label cleaning, frozen group-aware splits | 266/266 load; 47 orphans and 296 defect fibers set to ignore (training only); dev split 213/53 |
| 2 | Evaluator, error taxonomy, human ceilings from overlapping tiles | Found an EDA registration bug, which revealed 8 more overlaps (5 leaked). Splits regenerated **before** any training. Human F1 0.929, all disagreement is selection |
| 3 | nnU-Net data, pairing rule tested on GT pixels (R1), AxonPath and Cellpose zero-shot, B1 launched | The pairing rule is lossless on perfect pixels (F1 0.99994). AxonPath F1 0.784 |
| 4 | B1 trained (150 epochs, $7.69); all baselines scored; oracles run | B1 F1 0.904; separation 0.6% of errors |
| 5 | GO/NO-GO | **H1 = NO-GO** |

Details: [05](05-results.md). How each step works: [04](04-methods.md).

## 5. The oracle idea (the core experiment)

An **oracle** uses the ground truth to fix exactly *one* kind of error in a model's prediction. The score gain is the cost of that error kind.
- **O-sel:** delete predicted fibers that annotators left unlabelled.
  - Strict version: only those on a detected ring-like structure.
  - Broad version: all unmatched predictions on background.
- **O-sep:** keep the model's pixels but group them by the GT fibers. This removes merges, splits and mis-pairings.
- **O-bnd:** replace each matched fiber's pixels by the GT. This removes boundary error.
- **R1 (O-pair-GT):** feed *perfect* GT pixels to our pairing rule. This measures the rule itself.

## 6. Results vs assumptions (the honest scorecard)

| Assumption before the runs | What we found | Verdict |
|---|---|---|
| "Pairing/separation is the bottleneck" (from ASTIH's 0.79) | B1: 2 merges, 0 splits, 0 gap fibers in 1,619 GT fibers. Separation = **0.6%** of errors. O-sep gain −0.001 | **Wrong** for a properly sized nnU-Net on this data |
| The watershed line between touching rings is a guess that costs instances | R1: perfect pixels + our rule → F1 0.99994, g-MAE 0.0017 | **Wrong**: the rule is not an error source |
| Selection is a large hidden error source (EDA) | 78% of B1's errors are predictions on unlabelled structures. Humans disagree only on selection | **Right**, and it is the dominant cause |
| Human ceiling: myelin Dice ~0.93–0.96, g floor ~0.006 | Myelin Dice 0.951, g-MAE 0.0067, fiber F1 0.929 | **Right** |
| B1 Dice ≥ 0.95 axon / ≥ 0.88 myelin | 0.882 / 0.861. Mostly selection (unlabelled fibers count as FP pixels); without them: 0.948 / 0.908 | Expectation wrong; cause understood |
| Cellpose: higher precision, lower recall than B1 (as in ASTIH) | The opposite. Zero-shot Cellpose proposes a region for everything | **Wrong** (ours is zero-shot; ASTIH's was trained) |
| Early stopping saves 20–60% GPU | Never fired; all 150 epochs ran (the LR schedule ends at ~0 anyway) | No saving this time |
| H1 borderline; NO-GO possible | NO-GO by a wide margin | NO-GO |
| *(not anticipated)* | Grouping costs little in detection but **a lot in g-ratio**: O-sep cuts B1's g-MAE by 42%. Fibers whose predicted ring is broken (28%) carry 56% of the g error | New hypothesis **H5** |

## 7. Where we stand now

**Decided (D-27):**
- No nested decoder on the basis of H1.
- The project's main contribution is **C1**, the error decomposition: "on this data, fiber errors are selection, not pairing".
- The next method work is **C3**, which handles incomplete annotation.

**Open, and the owner's decision:**
1. **Budget.** $8.16 of the $10 is spent. C3 needs ≈ $31 of GPU time: out-of-fold predictions on all images, then a retrain. Cheaper variants are in [07](07-plan.md).
2. **H5.** "Grouping and ring topology drive g-ratio error". It is post hoc, so it needs a pre-written rule before it can justify reviving C2 as a *morphometry* method.
   - The cheapest first test uses no GPU: close broken rings and regroup B1's pixels instance-first.

**Not yet done:**
- Reproducing ASTIH TEM1 with our evaluator (T-17). This is the external check that our evaluator matches a published number.
- The write-up.

## 8. Easy to miss (the traps, collected once)

**Data and labels:**
- Masks are **uint16**. Saving a prediction as uint8 silently wraps IDs above 255 and breaks the X/X+1 pairing.
- Shapes are (H, W) = (3008, 4096) in numpy. The challenge text says "4096×3008", meaning W×H.
- IDs are not consecutive, and orphan IDs exist. Pair by X/X+1, never by rank.
- Label 1 must be ignored in **both** maps, and a prediction lying ≥ 50% on label 1 is ignored, not counted as FP. Otherwise a good model is punished for segmenting obvious fibers (752 of B1's 2,582 fibers lie there).
- Training-label cleaning (orphans and defects → ignore) is for **training only**. Evaluation uses the raw GT.
- In nnU-Net, the ignore label must be the **highest** label and be named `ignore`. Otherwise it silently becomes a real class.

**Splits and statistics:**
- Never split by tile or random image: overlapping tiles are the same axons. Splits are frozen; never regenerate them after results exist.
- nnU-Net makes its own random folds unless you give it `splits_final.json`.
- Images in an overlap group are not independent, so bootstrap by **group**.
- Marginal CIs (±0.04) are wider than H1's 0.03 threshold. Compare methods with **paired** bootstrap differences.

**Metrics:**
- **HD95 is dominated by extra or missing fibers**, not boundary placement. Human-vs-human myelin HD95 is 644 nm. Don't read HD95 as boundary accuracy.
- **Read g-MAE only with recall**, because it ignores misses.
- The **violation rate** is a topology diagnostic, not an error rate. The GT itself has 6.1%, and crops have 15%.
- `other_bg` false positives are **not** hallucinations: about half are real fibers (§5 of [05](05-results.md)).
- The human ceiling comes from only 129 fibers in small overlaps. Treat it as an indication.

**Engineering:**
- After moving a repo, the venv is not relocatable. Editable installs **and** `.venv/bin/*` launchers kept pointing at the old copy, and imports were silently stale. Check `which python` after activating.
- Never trust the exit code of a transfer piped through `tail`. A failed upload "succeeded".
- Count files after every download.
- nnU-Net's "New best EMA" log line is not the early-stopping rule. Replay the rule itself.

## 9. Glossary

| Term | Meaning |
|---|---|
| Fiber | One axon + its myelin ring (label X ∪ X+1) |
| g-ratio | √(axon area / fiber area): the equivalent-circle diameter ratio |
| Fiber F1@0.5 | Detection F1; a predicted fiber matches a GT fiber if their IoU > 0.5 |
| Selection error | A predicted fiber the annotators didn't label, or a labelled one the model skipped |
| Separation error | Merge (one prediction covers two fibers), split (one fiber in pieces), or a "gap" pseudo-fiber |
| Boundary error | The right fiber, with outlines slightly off (affects IoU 0.75, Dice, g) |
| Oracle | Fix one error type with the GT and measure the gain |
| Ceiling | The score of a second human annotator on the same tissue |
| B1 | nnU-Net ResEnc-M, 896×1152 patches, + the shared watershed pairing |
| Cellpose+B1 | Cellpose-SAM zero-shot regions, split into axon/ring by B1's pixels |
| C1 / C2 / C3 | Error-decomposition evaluation / nested decoder / incomplete-annotation learning |
| D-xx, T-xx | Decision and task IDs in [07](07-plan.md) |
