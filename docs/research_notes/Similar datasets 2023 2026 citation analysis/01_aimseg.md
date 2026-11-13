# AimSeg corpus-callosum TEM (PLoS Comput Biol 2023; Zenodo 8351731 / 6327209 / 8021007)

## Introducing paper (read in full, PMC10691719)
Carrillo-Barbera, Rondelli, Morante-Redolat, Vernay, Williams, Bankhead. PLoS Comput Biol 19(11):e1010845, 17 Nov 2023. Edinburgh (Williams lab) + Valencia.
- **Data:** adult mouse CC, **remyelinating after a unilateral toxin-induced demyelinating lesion** + healthy control. JEOL JEM-1400 Plus, GATAN OneView, 7.1k x, 4096x4096, **2.1046 nm/px** (8.62 x 8.62 um) - acquisition very close to ours (2.152 nm/px). Downsampled x4 (8.42 nm/px) for the tool.
- Splits: classifier training 5 images from 1 mouse; **validation GT 5 images / 4 mice**; **control GT 3 images / 1 healthy mouse**. One expert annotator (QuPath).
- **Labels:** hierarchical instances per fiber: fiber ⊃ inner region (axon + inner tongue) ⊃ axon; plus a semantic mask (axon / inner tongue / compact myelin). **Pairing is by construction (hierarchy)**.
- **Selection:** "Technical artefacts, **degraded myelin debris and degenerated dark axons were not included**" in GT. Image borders removed for morphometry. -> same selective-annotation convention as ours.
- **Method:** ilastik random-forest pixel + object classifiers, Fiji rule pipeline (inner region -> seeded expansion to fibers -> axon instances), optional user-assisted mode.
- **Metrics:** object F1 averaged over IoU 0.5-0.9; Lin's CCC + Bland-Altman for fiber area, "myelin g" (inner region / fiber) and "axon g" (axon / fiber); equivalent-circle diameters.
- **Results (automated -> supervised):** mean F1 inner region 0.83 -> 0.87, fiber 0.85 -> 0.88, axon 0.71 -> 0.83. At IoU 0.5: 15% FN, 5.8% FP (auto). Fiber area CCC 0.999. **Myelin g CCC 0.83, bias -0.01, LoA -0.07..+0.06; axon g CCC 0.75, LoA -0.13..+0.12.** Control: similar (myelin g CCC 0.86).
- Key biology/measurement point: the **inner tongue biases the classic g-ratio**; two fibers with different tongues can have the same myelin g.
