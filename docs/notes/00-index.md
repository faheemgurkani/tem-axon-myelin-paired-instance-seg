# Research Notes — Index

Living notes for the project. Read in order the first time; afterwards jump to what you need.

| # | Note | What it holds |
|---|---|---|
| 01 | [Challenge spec](01-challenge-spec.md) | **Verified** task, data, label encoding, metrics, submission and dates, plus site inconsistencies |
| 02 | [Problem framing](02-problem-framing.md) | Plain-language explanation, ML framing, relation to the seizure work, critique of the plan |
| 03 | [Literature review](03-literature-review.md) | SOTA tools, surveys, transferable techniques, gaps (preserved deep-research brief + corrections) |
| 04 | [Baselines](04-baselines.md) | Ranked baselines B1–B5, shared pairing, challenge-mode recipe |
| 05 | [Evaluation framework](05-evaluation-framework.md) | Official-metric replica, diagnostics, morphometry, split protocol |
| 06 | [Research directions](06-research-directions.md) | Nested-instance method, topology losses, SSL, label efficiency, venues |
| 07 | [Datasets](07-datasets.md) | Challenge data + external datasets for pretraining and validation |
| 08 | [Open questions & risks](08-open-questions-and-risks.md) | Organizer questions, risks, resolved items |
| 09 | [Research plan](09-research-plan.md) | Phases P0–P5, schedule, log |

## Deep-research outputs (2026-11-11)

| Item | What it holds |
|---|---|
| [**Report: TEM axon myelin segmentation review**](../reports/TEM%20axon%20myelin%20segmentation%20review.md) | **Main synthesis**: SOTA, a 52-claim audit of the prior review, datasets and overlap, baselines B0–B5, evaluator defaults, the research gap and hypotheses H1–H4 |
| [Research notes](../research_notes/TEM%20axon%20myelin%20segmentation%20review/) | Source notes: domain methods, datasets and provenance, instance techniques, topology/losses/pretraining, evaluation methodology |
| [Prior review (verbatim)](../references/prior-literature-review-2026-11-10.md) and [search log](../references/prior-review-search-log-2026-11-10.md) | First deep-research pass as pasted |

Papers (PDF/text) and cloned repos are kept locally and git-ignored, in `docs/references/papers/` and `external/`.

Raw snapshot of the challenge website content: [`../references/challenge-site-snapshot/`](../references/challenge-site-snapshot/)
(source repo `alexandrojim/tem-challenge` @ `ec9b8e3`, crawled 2026-11-10).

**Precedence:** 01 (verified from the site) > the 2026-11-11 report (primary-source verified) > 03 (secondary review). Record new evidence in 08.
- [10 — EDA of the training set](10-eda.md)
- [11 — Deep EDA: label quality, image forensics, FP patterns, leakage](11-deep-eda.md)
