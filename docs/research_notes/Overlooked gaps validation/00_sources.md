# Sources and evidence: overlooked-gaps validation (15 Nov 2026)

Report: `docs/reports/Overlooked gaps validation.md`.

## Web (read or searched 15 Nov)

| Source | Status | How read | Key evidence |
|---|---|---|---|
| Parikh, Frank, Das, Feragen, "Towards Fairness under Label Bias in Image Segmentation" ([arXiv 2605.06891v2](https://arxiv.org/html/2605.06891v2), 26 Sep 2026; code [ADE-17/label-bias-segmentation](https://github.com/ADE-17/label-bias-segmentation)) | arXiv; a third-party site lists NeurIPS 2026, **not stated on the paper** | HTML, first ~40k characters verbatim, the rest via summary (unverified) | Bias simulated as dilation, erosion and boundary deformation per subgroup; datasets CelebAMask-HQ, PhC-U373, IMA++, ISIC. Audit = confident learning on cross-fitted predictions; "proxy estimates". Mitigation: FiLM subgroup-conditioned decoder; "expresses all predictions in the reference annotation style". No object omission, no morphometry |
| Zepf et al., "That Label's Got Style" ([arXiv 2303.15850](https://arxiv.org/abs/2303.15850)) | ICLR 2023 | search result | Annotations modelled conditioned on a discrete label-style variable; condition on the target style at inference |
| Deng, Miao, Bagheri, Peterson, Khadra, MicroStructFormer ([bioRxiv 10.64898/2026.05.26.727755](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v1.full)) | preprint, May 2026 | full text in Chrome (bot check cleared itself after a 20 s wait; nothing solved) | Classes: axon, myelin, mitochondria, PAS, **abnormality** (uncompacted membrane). Myelin-guided watershed + geodesic assignment. Exclusive vs inclusive fiber. "a slight divergence was observed in the abnormality class, particularly in the most severe" genotypes. Labels by "iterative, expert-guided annotation" of model predictions. Data and weights "being organized for public release" |
| Oakden-Rayner, Dunnmon, Carneiro, Ré, "Hidden stratification…" ([doi 10.1145/3368555.3384468](https://www.doi.org/10.1145/3368555.3384468)) | ACM CHIL 2020 (origin of the concept) | search result | Performance gaps > 20% on clinically important subsets |
| Bissoto et al., "Subgroup Performance Analysis in Hidden Stratifications" ([arXiv 2503.10382](https://arxiv.org/pdf/2503.10382)) | MICCAI 2025 | search result | Label-free subgroup discovery reveals larger disparities than metadata-based analysis |
| SIPAIM 2026 challenge site ([main page](https://alexandrojim.github.io/tem-challenge/)) | live page, 15 Nov | fetched; winners, submission and data subpages 404 | Deliverables: predictions, a 2–4 p PDF, code. No GT, prediction or code release stated. "Winners will be announced at SIPAIM 2026"; three unverified named entries shown |
| Fields & Dutta, G-ratio commentary ([PMC12140491](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12140491/)) | ASN Neuro 2025 | search result | EM sampling is "inadequate and inappropriate"; the slope is a sampling and measurement artifact |
| Nerve-morphometry bias levels ([Springer s004290100188](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s004290100188.pdf)) | older journal paper | search result only | Inclusion and exclusion rules for fiber profiles are a named level of sampling bias |
| Efficient end-to-end learning for cell segmentation… ([PMC9981753](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9981753/)) | 2022–23 | search result | Annotators did not label dying, abnormal-looking cells; avoided colony centres |
| Imaging-mass-cytometry missing-label study (2025) | search snippet | — | Missing cells simulated by random erasing only |
| Overlapping-tile agreement | 1 search | — | No source found describing agreement estimated from overlapping tiles |

**Not done:**
- No deep search for a recall-aware morphometric metric.
- The Parikh paper's results and limitations sections were read only as a summary.

## Local evidence used

- `docs/research_notes/Phase 0b evidence/07_external_datasets.md` §4: the by-eye composition of our candidates (~60% abnormal: onion or thick myelin, large axons with thin myelin, oblique). TEM1 and AxonPath exclusions. The TEM1 proxy cluster 2 has 19% of fibers with g > 0.85.
- `docs/notes/05-results.md` §5: B1's `other_bg` gallery, where ~7 of 16 are real unlabelled fibers, "mostly thin myelin". The detector misses thin myelin.
- `data/processed/eda/unl_cands.csv`: the columns hold the inner diameter only (`d_nm`), with **no outer boundary, so no g**.
- `docs/research_notes/Phase 0b evidence/06_g_effect_sizes_B5.md`: the 0.03 yardstick and the "biased low" hypothesis (one-sided, unsupported).
- `docs/research_notes/Phase 0b evidence/10_expert_review_protocol.md`: category 2 = abnormal (thin, redundant/onion, degenerating, oblique). This is usable for the class option and for outline tracing.

## Addendum sources (15 Nov)
- Parikh et al. abstract page ([arXiv 2605.06891](https://arxiv.org/abs/2605.06891)): Comments "The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS) 2026"; no journal reference; v1 7 May, v2 26 Sep 2026. Full HTML v2 read verbatim (subagent, all chunks).
- [OpenReview search](https://openreview.net/search?term=Towards%20Fairness%20under%20Label%20Bias%20in%20Image%20Segmentation), read in Chrome 15 Nov: entry "Published: 31 Dec 2026 … CoRR 2026"; no NeurIPS 2026 decision. The same search shows Zepf et al. as "ICLR 2023 poster" (confirms that venue).
- [SortedAP, arXiv 2309.04887](https://arxiv.org/pdf/2309.04887); Metrics Reloaded / pitfalls ([Nat Methods 2024](https://www.nature.com/articles/s41592-023-02151-z.pdf)); ensemble nerve morphometry with KS tests ([Electronics 2022](https://mdpi.com/2079-9292/11/14/2277/htm)); [gACSON](https://arxiv.org/html/2112.06476v2); PQ: local `docs/references/papers/kirillov2019_panoptic_quality_1801.00868.pdf`.
