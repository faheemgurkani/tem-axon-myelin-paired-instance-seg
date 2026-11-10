# Evaluation Metrics & Submission Protocol

## 1. Evaluation Metrics & How They Are Computed

All test-set submissions will be evaluated across three complementary criteria designed to measure spatial overlap, physical boundary accuracy, and per-fiber biological morphometry for the paired **Axon (`X`)** and **Myelin Sheath (`X + 1`)** instances.

> **Artifact Masking & Permutation Invariance:** Pixels annotated as **Label `1` (Artifacts)** in the manual ground truth are ignored and excluded from all metric computations. Furthermore, the specific even/odd integer IDs assigned to predicted fibers do not need to match the exact integer IDs of the ground truth—predicted fiber pairs `(X, X + 1)` are automatically matched to ground-truth fiber pairs via maximum Intersection-over-Union (IoU), provided each predicted axon (`X`, even $\ge 2$) is consecutively paired with its enclosing myelin sheath (`X + 1`, odd $\ge 3$).

### 1.1 Dice Similarity Coefficient ($\text{DSC}$)
The Dice Similarity Coefficient quantifies spatial overlap for both the **Axon** compartment (all even instance labels $X \ge 2$) and the **Myelin** compartment (all odd instance labels $X + 1 \ge 3$) across matched instances and global foreground masks between predictions ($P$) and ground truth ($G$):

$$\text{DSC}_c = \frac{2 \vert{}P_c \cap G_c\vert{}}{\vert{}P_c\vert{} + \vert{}G_c\vert{}}$$

Both compartment scores ($\text{DSC}_{\text{Axon}}$ and $\text{DSC}_{\text{Myelin}}$) and their macro-average ($\text{DSC}_{\text{Mean}}$) are reported across all test micrographs.

### 1.2 95th Percentile Hausdorff Distance ($\text{HD}_{95}$)
To evaluate contour localization along the inner axonal boundary and the outer myelin boundary while suppressing sensitivity to isolated single-pixel outliers, we compute the 95th percentile Hausdorff Distance ($\text{HD}_{95}$) for the **Axon** and **Myelin** boundaries:

$$\text{HD}_{95}(P_c, G_c) = \max \left\{ P_{95}\left(\left\{\min_{g \in \partial G_c} \Vert{}p - g\Vert{}_2 : p \in \partial P_c\right\}\right), \; P_{95}\left(\left\{\min_{p \in \partial P_c} \Vert{}g - p\Vert{}_2 : g \in \partial G_c\right\}\right) \right\} \times s$$

where $\partial P_c$ and $\partial G_c$ denote the boundary contours of the predicted and ground-truth masks, and $s = 2.152\text{ nm/pixel}$ converts pixel distances into physical nanometer ($\text{nm}$) units.

### 1.3 Per-Fiber Morphometric Accuracy ($g$-Ratio Error)
In white-matter morphometry, the gold-standard indicator of myelination is the **$g$-ratio**, defined for each individual myelinated fiber as the ratio of the inner axonal diameter to the total outer fiber diameter (axon plus surrounding myelin sheath):

$$g_X = \frac{D_{\text{axon}}(X)}{D_{\text{fiber}}(X \cup X+1)}$$

Because each myelinated fiber is explicitly segmented as a linked **`(X, X + 1)` instance pair**, our automated evaluation script computes per-fiber morphometry directly from the instance masks by:
1. Matching each predicted fiber pair `(X, X + 1)` to its corresponding ground-truth fiber pair via IoU on the complete fiber mask ($X \cup X+1$).
2. Computing the equivalent circular inner axonal diameter $D_{\text{axon}}(X) = 2\sqrt{A_{\text{axon}}(X) / \pi}$ from label `X` and the outer fiber diameter $D_{\text{fiber}}(X \cup X+1) = 2\sqrt{A_{\text{fiber}}(X \cup X+1) / \pi}$ from the union of labels `X` and `X + 1` via `skimage.measure.regionprops`.
3. Calculating the **Mean Absolute Error ($\text{MAE}_g$)** between predicted and ground-truth fiber $g$-ratios across all matched fibers:

$$\text{MAE}_g = \frac{1}{N_{\text{matched}}} \sum_{i=1}^{N_{\text{matched}}} \left\vert{} g_i^{\text{pred}} - g_i^{\text{gt}} \right\vert{}$$

---

## 2. Required Submission Deliverables

To qualify for the final ranking and SIPAIM award, each registered individual or team must upload a **single `.zip` archive** named:

`TeamOrLastName_Institution_SIPAIM2026.zip`

Your archive must follow this exact directory structure:

```text
TeamOrLastName_Institution_SIPAIM2026.zip
├── method_description.pdf
├── code/
│   ├── README.md
│   ├── requirements.txt (or environment.yml)
│   ├── train.py
│   ├── predict.py
│   └── checkpoints/ (or download script/link for trained weights)
└── predictions/
    ├── image_001.tif (or .png saved as uint16, matching test set filenames)
    ├── image_002.tif
    └── ...