# AimSeg -> AxonPath reuse

- AxonPath datasets (Zenodo 23084690, 1 Oct 2026, CC-BY): 47 images, 9,899 fibers. CorpusCallosum_Mus_TEM_000: 20 images, 2,074 fibers, 1.75-2.11 nm/px, **8 of them are AimSeg images** (SOURCES.csv). Annotation per fiber: Fibre ⊃ InnerCylinder ⊃ Axon (QuPath GeoJSON); training masks 0 bg / 1 myelin / 2 inner tongue / 3 axon + fiber instance labels. Only fibers inside an annotated ROI are labelled.
- AxonPath model (code, Sep-Oct 2026): one U-Net, three heads (semantics + fibre distance + axon distance); inner cylinder assigned to the fibre containing >= 90% of it (rule) - see report B.
- No paper yet -> no reported numbers on the CC subset. 0 citers.
- MicroStructFormer (bioRxiv 2026) lists AimSeg among public EM data for Mask2Former domain pre-training; reports no AimSeg-specific results.
