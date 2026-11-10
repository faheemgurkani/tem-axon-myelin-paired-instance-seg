# 02 — Problem Framing (for an ML engineer, not a neuroscientist)

## Field

- **Discipline:** neuroscience → microscopic neuroanatomy (**neurohistology**).
- **Sub-area:** **myelin and white-matter biology**: how nerve fibers get their insulation (myelin), how healthy
  it is, and how diseases (multiple sclerosis, ALS, leukodystrophies) damage it and how remyelination repairs it.
- **Our side:** **bioimage analysis**, i.e. computer vision that measures biological structures in microscope images.

This is about the **physical structure** of nerve fibers, not brain activity. That separates it from our
seizure-onset work, which is **electrophysiology** (EEG time series).

## The task in plain terms

A TEM image of corpus callosum tissue looks like a cross-section of a bundle of cables. Each insulated cable
(myelinated fiber) appears as:
- a **light disk**, the axon (the wire)
- surrounded by a **dark ring**, the myelin (the insulation)

For every cable, we must output **one disk and the one ring that belongs to it**, with linked IDs (`X`, `X+1`).
From these we compute the **g-ratio = axon diameter / (axon + myelin) diameter**. Biologists track g-ratio
to tell whether insulation is thinning (disease) or recovering (therapy). Healthy CNS values are roughly 0.6–0.8.

## ML framing

| Aspect | Formulation |
|---|---|
| Problem type | 2D **instance segmentation** with a **hierarchical / nested** constraint: each instance has two parts, the inner disk and the outer annulus |
| Structural prior | Axon ⊂ fiber; myelin = fiber \ axon; myelin is an annulus (Betti number β1 = 1); fibers touch but must not merge |
| Inputs | Very large grayscale images (4096×3008), 266 labeled, tiled training |
| Outputs | uint16 instance map with even/odd pairing |
| Hard parts | Touching rings (instance separation); a 1–2 px inner-boundary shift moves g-ratio; small axons; unmyelinated axons (negatives); border-cut fibers; artifacts |
| Metrics | Dice (easy, everyone similar), **HD95 (boundary)** and **g-ratio MAE (morphometry)**, which decide the ranking |

The closest ML analogues are nucleus instance segmentation (HoVer-Net, StarDist, Cellpose) and topology-aware
segmentation (clDice, Betti matching, containment constraints). Here, though, every object is a **two-part object**.

## Relation to the seizure-onset work

| | Seizure onset (current) | This project |
|---|---|---|
| Data | EEG time series | 2D EM images |
| Field | Electrophysiology | Neurohistology / white matter |
| Task | Temporal classification / forecasting | Structured instance segmentation |
| Shared lesson | **Leakage-free splits** (LOSO by patient) | Split by **image/animal**, never by tiles of the same image |
| Shared lesson | Event-level vs epoch-level metrics | Instance-level vs pixel-level metrics |

## Difficulty and feasibility

- **Feasible.** Standard tooling (nnU-Net, Cellpose, AxonDeepSeg) gives a strong baseline quickly. The image contrast is high and the SNR is good.
- **Hard to win or publish on Dice alone.** The research value lies in nesting/topology, boundary precision
  (g-ratio) and generalization. These are the two gaps the organizers name.
- **Compute:** large images, but 2D. A single 24 GB GPU is sufficient for nnU-Net 2D.

## Critique of the "baselines first" plan (keep in mind)

1. **Time-box the baselines.** At most 3–5 models. Don't let the baseline phase consume the project.
2. **Compare like with like.** Semantic models don't output pairs. Apply the *same* pairing post-processing to every
   baseline and score all of them with *our replica of the official metric*. **Build the evaluator before any training.**
3. **Avoid strawmen.** Weak baselines inflate our gains and reviewers notice. nnU-Net self-tunes, which protects us here.
4. **Avoid leaky validation.** Split by whole image, and by animal or block if the metadata allows.
5. **Error analysis decides the method, not intuition.** Count failures by type and by fiber-size bin:
   - merged neighbouring fibers
   - missed small axons
   - inner boundary shifted by 1–2 px (g-ratio)
   - fibers cut off at the image border
   - unmyelinated axons labelled as fibers
   - pairing violations (axon without ring, ring with 0 or ≥2 axons)

   This breakdown becomes the motivation section of the paper.
6. **Winning and novelty diverge.** Challenge winners are usually tuned nnU-Net + post-processing + TTA + ensembling.
   Novelty requires a topology-aware / nested-instance contribution. Track both, and know which one each experiment serves.
7. **Reproducibility from day 1:** `train.py`, `predict.py`, env file and weights. The challenge requires all of them.
