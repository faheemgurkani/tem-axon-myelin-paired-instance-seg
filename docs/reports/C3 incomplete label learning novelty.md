# Is C3 new? Learning with incompletely annotated fibers (14 Nov 2026)

**C3** (D-18, T-15′) means: in the training labels, ignore high-confidence ring candidates that the annotators left unlabelled, then retrain B1.

**The question (TODO, Q5 item 1):** is this novel? If not, how was it done before? Does any method handle objects left out **on purpose** (our abnormal fibers), and not only objects missed at random?

Notes and evidence: `docs/research_notes/C3 incomplete label learning novelty/00_papers_read.md`.

## Verdict: the mechanism is not new; the selection-aware framing and its evaluation are new for this domain

1. **The mechanism already exists.**
   - **nnU-Net's ignore label** (Gotkowski et al. 2024) computes the loss only on labelled pixels. Its documentation lists masking out faulty reference regions as a use case, so **D-18 is essentially a built-in feature.**
   - **Engelson et al.** (LNQ 2023, *MELBA* 2024) is the closest prior work. Their annotators labelled only "clinically relevant", enlarged lymph nodes, a deliberate selection like ours.
     - They soften the Dice loss wherever an atlas predicts a node but no label exists.
     - Gain: Dice +0.024 and node recall +0.03, with no significant loss of precision. This is modest, and smaller than the gains from oversampling the fully labelled cases.
2. **PU learning handles missed objects, but has been tested almost only on random omission.**
   - Zhao et al. (*MELBA* 2023): PU cell detection, with labels deleted **uniformly at random**.
     - Large F1 gains against "unlabelled = negative" (for example 0.10 → 0.52 at one label per patch).
     - One easy-cells-only test also helped.
   - Natural-image sparse-annotation detection (SparseDet, Calibrated Teacher, the PU detector) also thins COCO and VOC labels at random.
   - Only one variant found ties missingness to the object, and only to its size: BMVC 2021, large objects only.
3. **Selective omission is documented, but nobody corrects for it in segmentation.**
   - Komura et al. (*Patterns* 2023) show pathologists miss atypical-looking cells, "an inherent morphological bias". Their remedy is a new labelling protocol, not a training method.
   - Omission studies (Vădineanu et al., MIDL 2022) simulate random omission only.
   - The only formal method for selection that depends on the label (BiasCorr, arXiv 2023) is for tabular classification.

**So C3 cannot claim a new loss.** It can claim three things no paper found here does:
- **(a)** characterise selective omission in EM: which fibers are left out (abnormal or thin myelin) and how many (about 4–8%, and 78% of B1's errors);
- **(b)** measure its effect on detection **and on per-fiber g-ratio**;
- **(c)** evaluate recovery against a set where a missing label is **known** to be a real fiber: our overlap-disagreement fibers, where the same fiber is labelled in one tile and background in the other. The other papers either make the test labels complete or measure on random deletions they simulated themselves.

## What this changes in T-15′

- **Implement D-18 with nnU-Net's ignore label.** Our masks already use label 1 as "ignore". Cite Gotkowski 2024 and do not present the loss as ours.
- **Add baselines that answer "how was it done before":**
  1. a hard ignore (D-18 as planned);
  2. LNQ-style soft down-weighting (weight about 0.75 on candidates);
  3. a PU-style loss on fiber pixels (nnPU with the class prior tuned on recall, as Zhao et al. do);
  4. a **random-ignore control** with the same pixel budget. This is what shows that targeting *selected* fibers matters, which none of the papers above can show.
- **Set expectations.** The literature suggests modest gains (LNQ: +2–3 points recall). Our H3 asks for ≥ 10 points on overlap-disagreement fibers, so it may be too optimistic.
  - Do not change H3 now; record the outcome against it, per the plan rules.
  - Official metrics will still score recovered unlabelled fibers as false positives, so precision may fall. That is why the overlap-disagreement evaluation is the primary one.
- **Risk (Bukas et al., MICCAI-W 2026):** ignore masks built from a model can add bias.
  - Build candidates from ring evidence first, and use the out-of-fold model only as a filter.
  - Score against human labels only.
- **Paper framing:** C3 is "selective-omission-aware training and its evaluation in EM, with a known-positive test set", not a new loss. C1's breakdown (selection = 78% of errors) is what motivates it.

## Not covered
- The 2024 *J Imaging* omission-correction paper by Vădineanu et al. (DOAJ returned 403).
- ReCo-I2P (MICCAI 2025) and the BMVC 2021 large-objects paper: titles and snippets only.
- No 2025–26 paper at MICCAI, MedIA, TMI, CVPR or NeurIPS was found on PU or MNAR segmentation. Search coverage of the 2026 proceedings was only partial.
