# TEM Imaging Dataset of the Mouse Corpus Callosum

## 1. Dataset Overview

This dataset comprises a curated collection of **400 minimally processed, high-resolution Transmission Electron Microscopy (TEM) images** acquired from the corpus callosum (white matter) of mouse (*Mus musculus*) brain tissue at the **Paris Brain Institute (Institut du Cerveau - ICM)**.

The dataset was assembled to support the development and benchmarking of deep-learning and image-processing models for the automatic paired instance segmentation of axons and their surrounding myelin sheaths.

| Item | Specification |
| :--- | :--- |
| **Imaging Modality** | Transmission Electron Microscopy (TEM) |
| **Anatomical Region** | Corpus callosum (white matter) |
| **Species** | Mouse (*Mus musculus*) |
| **Total Images** | 400 (266 train + 134 test) single-channel grayscale micrographs |
| **Image size** | 4096 x 3008 pixels |
| **Annotation Format** | **Paired Instance Masks (`uint16`):** Individual Axons (`X`) and paired Myelin sheaths (`X + 1`), plus Artifacts (`1`) |
| **Intended Use** | Supervised, semi-supervised, or self-supervised instance segmentation training & evaluation |

---

## 2. Instance Annotation Encoding & Topological Nesting

At TEM resolution, each myelinated fiber exhibits high contrast between the lipid-rich, electron-dense myelin sheath (dark) and the enclosed axoplasm (lighter). Ground-truth masks are stored as 16-bit unsigned integer (`uint16`) arrays using the following paired instance label encoding:

| Label ID | Class / Instance Role | Morphological Description |
| :--- | :--- | :--- |
| **`0`** | **Background / Extra-axonal Space** | Surrounding neuropil and unmyelinated tissue matrix between fibers |
| **`1`** | **Artifacts (Ignored)** | Histological or imaging artifacts (excluded from training loss and test evaluation) |
| **`X` (Even integer $\ge 2$)** | **Axon Instance (e.g., `14`)** | Cross-section of an individual axoplasm compartment appearing as a lighter, roughly circular internal region |
| **`X + 1` (Odd integer $\ge 3$)** | **Paired Myelin Instance (e.g., `15`)** | Concentric, electron-dense (dark) ring of tightly wrapped membrane lamellae enclosing Axon `X` |

> **Paired Topological Rule:** Because each myelin ring wraps around a single axon, every myelinated fiber is encoded as a strictly linked **`(X, X + 1)` instance pair**, where `X` is an even integer ($\ge 2$) representing the axon and `X + 1` is the consecutive odd integer representing its surrounding myelin sheath (for example, if an axon is labeled `14`, its myelin sheath is labeled `15`). Pixels labeled **`1` (Artifacts)** are not considered during scoring.

---

## 3. Image Acquisition & Technical Specifications

All micrographs were acquired on a transmission electron microscope operated in imaging mode using an AMT NS12 camera system. The acquisition parameters below are read directly from the embedded metadata.

| Parameter | Value |
| :--- | :--- |
| **Microscope Mode** | TEM — Imaging mode |
| **Accelerating Voltage** | 100 kV |
| **Direct Magnification** | 3000x |
| **Camera System** | AMT NS12 (AMT Camera System) |
| **Pixel Calibration** | 0.002152 µm/pixel ($\approx$ 2.15 nm/pixel) |
| **Embedded Scale Bar** | 1 µm ($\approx$ 465 pixels at stated calibration) |
| **Exposure & Frame Averaging** | 480 ms × 4 standard frames (averaged for noise reduction) |
| **Sensor Resolution & Binning** | Binning = 1, Gain = 1 (full native sensor resolution, no on-sensor pooling) |
| **Radiometric Response** | Gamma = 1.00 (linear response), Normal contrast, Sharpening = None |

### Key Image Characteristics for Modeling
* **Nanometer-Scale Spatial Resolution:** At $\approx 2.15\text{ nm/pixel}$, the spatial resolution resolves individual myelin lamellae as well as the sharp boundary between the inner myelin sheath and the axoplasm.
* **Quantitative Intensity Fidelity:** A linear gamma of $1.00$ with no artificial sharpening preserves the quantitative relationship between pixel intensity and tissue electron density.
* **High Signal-to-Noise Ratio:** Frame averaging across $4$ exposures of $480\text{ ms}$ yields clean edges along the inner axonal and outer myelin boundaries.