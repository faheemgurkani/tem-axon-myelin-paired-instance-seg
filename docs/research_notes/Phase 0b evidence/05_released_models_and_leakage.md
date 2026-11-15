# Released models: availability and what they trained on (gap C3), checked 15 Nov 2026

Sources: GitHub API (releases, issues, scripts), the Zenodo API, local clones `external/ASTIH` @ 089896e and `external/axonpath` @ 6d633bf.

| Model | Where | Downloads? | Trained on | Fair test set for us |
|---|---|---|---|---|
| **ASTIH TEM1** = `axondeepseg/default-TEM-model` r20240403 (nnU-Net v2, 3 of 5 CV folds + `fold_all`; `_light` zip 249 MB, `_ensemble` 748 MB) | `scripts/dataset_list.json` in ASTIH points here | **Yes**: light zip downloaded to `data/external/models/` (249,263,424 B) | All TEM1 subjects **except `sub-nyuMouse26`**: `nnunet_scripts/prepare_tem_dataset.py` puts that subject in `imagesTs` ("excluded for model evaluation"). Trained late 2023, before ASTIH's 2-px re-annotation was released, so the label version may differ from DANDI 0.250512.1625 | `sub-nyuMouse26` only. **Leakage-free** |
| **ASTIH TEM2** = `axondeepseg-pipeline/model_seg_unmyelinated_tem` r20240708-stanford (5-fold nnU-Net ensemble) | `dataset_list.json` | Not yet downloaded (only needed for T-17 on TEM2) | The 10 annotated DANDI images. The split file `scripts/nnunet/train_test_split_SRF.json` lists train and test samples **from the same mice** (sub-370…375). Test labels were released later (May 2025) as `data_axondeepseg_srf_testing` | The external test zip (downloaded: 21 ROI images from 10 mice at 4.93 nm/px, each with axon, myelin, axonmyelin and uaxon masks). **Probably leakage-free but in-animal** (C4 confound 3). Not confirmed that r20240708 never saw those ROIs |
| `model_seg_unmyelinated_tem` **v3.0.0** (Jun 2026) | same repo | — | TEM2 **plus an unpublished dataset** | **Never use on TEM2** |
| **AxonPath `em_cns-0.1.0`** (8 nm/px) | Zenodo 22982902 (CC-BY, 26 Sep 2026; "isDerivedFrom" the dataset record 23084690) | Yes (17.8 MB). Our zero-shot baseline used AxonPath's bundled EM weights | **Tile-level** split (`data_preparation.py`: 80/10/10 of shuffled tiles; the docstring calls it "a known limitation"). Every CC TEM image contributed training tiles | **None on AxonPath CC TEM.** Use that dataset only for GT-only analyses and for scoring *our* models |

## Consequences
- **T-17 (TEM1):** reproduce on `sub-nyuMouse26` with the `default-TEM-model`, the ADS centroid watershed (`04_axondeepseg_tool_review.md`), and an explicit IoU threshold of 0.5.
  - The released `_light` model may not be the exact checkpoint behind ASTIH Table 4 (1000 epochs, 5-fold). **The ±0.01 / ±0.03 tolerance in T-17 decides whether it matches.**
- **TEM2:** treat it as an in-animal benchmark. Scores are optimistic (C4).
- **AxonPath CC TEM** can serve:
  - the completeness audit;
  - the pairing metric and H5 checks on GT;
  - external scoring of **our** B1-family models.

  AxonPath's own model must not be scored on it.
- **Our existing AxonPath zero-shot result on the challenge data is unaffected.** It is a different dataset, so there is no leakage.
