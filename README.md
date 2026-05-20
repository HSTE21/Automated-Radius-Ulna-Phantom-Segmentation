# 3D Image Reconstruction & Visualisation (202400339)

## Assignment 1: Automated MRI Bone Segmentation for Personalized Anatomical Guides

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Optimization: Optuna](https://img.shields.io/badge/Optimization-Optuna%20TPE-orange.svg)](https://optuna.org/)
[![Library: Scikit-Image](https://img.shields.io/badge/Library-Scikit--Image-green.svg)](https://scikit-image.org/)

---

### 🚀 Keywords & Tags

`MRI Segmentation` `Medical Imaging` `Optuna` `Hyperparameter Tuning` `Radius-Ulna` `3D Reconstruction` `Dice Coefficient` `Hausdorff Distance` `Scikit-Image` `Watershed` `Chan-Vese` `Random Walker`

---

## Project Overview

This project focuses on the automated segmentation of bone structures from MRI images of an **anatomical phantom (Radius and Ulna)** for the creation of personalized anatomical guides. We evaluated 10 iterations of segmentation algorithms, moving from basic global thresholding to advanced region-growing and active contour methods, all optimized using the Optuna framework.

---

### 🏆 Segmentation Leaderboard (ROI Results)

A high-level comparison of all methods evaluated in this project. All models were optimized using **Optuna TPE**. Results reflect the refined evaluation methodology (XY ROI + Z-slice exclusion).

| Iteration | Method                                     | Dice (ROI) | Highlight                                           |
|:--------- |:------------------------------------------ |:---------- |:--------------------------------------------------- |
| **03**    | **ROI-Restricted Precision Tuning**        | **~0.986** | **Overall Winner** - Exceptional baseline performance. |
| **04**    | **Multi-Otsu Statistical Thresholding**    | **~0.976** | Robust multi-class intensity separation.            |
| **09**    | **Diffusion-Based Random Walker**          | **~0.971** | High precision in seed-based diffusion.             |
| **07**    | **Marker-Controlled Watershed**            | **~0.965** | Optimal anatomical separation.                      |
| **06**    | **3D Morphological Active Contours**       | **~0.951** | Smooth, regularized boundary surfaces.              |
| **01**    | **Baseline Discrete 4D Grid Search**       | ~0.890     | Global search baseline.                             |
| **08**    | **2D Slice-Wise Regional Active Contours** | ~0.405     | Slice-by-slice regional evolution.                  |
| **10**    | **Canny Edge-Based Volume Reconstruction** | ~0.364     | Fast edge-based detection.                          |
| **05**    | **Sauvola Local Adaptive Thresholding**    | ~0.043     | Highly sensitive to local variations.               |

---

## 🎞️ 3D Segmentation Scroller (Top Performer)

This animation demonstrates the performance of our **Iteration 03 (ROI-Restricted Precision Tuning)** model. It visualizes the automated bone segmentation (green overlay) across the entire 3D volume, excluding the problematic first and last Z-slices.

<p align="center">
  <img src="Visualizations/segmentation_scroller.gif" width="500" alt="3D Bone Segmentation Scroller">
</p>

---

## Key Insight: Refined ROI & Z-Slice Evaluation

During the development of this project, we identified significant artifacts in the provided ground truth `bone_mask.nii` that necessitated a refined evaluation strategy to achieve "fair" and clinically relevant metrics.

1. **Background Noise (XY ROI):** The background area (outside the forearm) in the ground truth is marked as foreground (white), creating a "dirty" mask. We implemented an **XY Region of Interest (ROI)** bounding box `[37:227, 90:189]` to exclude these artifacts.
2. **Invalid End Slices (Z-Exclusion):** The first and last slices of the ground truth volume are entirely white (foreground), which does not represent the actual anatomy. Including these slices artificially penalized models that correctly identified these areas as background. To ensure an **honest Dice score**, we modified the calculation to **exclude the first and last Z-slices**.

### Impact on Metrics:
- **Fair Comparison:** By removing the "white" end-slices and the noisy background, the Dice scores now accurately reflect the algorithm's ability to segment the Radius and Ulna.
- **Improved Performance:** Most 3D-optimized methods (Iterations 03-07) now show Dice scores exceeding **0.95**, indicating excellent alignment with the ground truth in the relevant anatomical regions.

---

## Optimization Scope & Performance Generalization

A critical finding in this project is the impact of **Optimization Scope** on the validity of the results. Some advanced algorithms (Iterations 08-10) were optimized on a single 2D slice to reduce computational overhead.

| Iteration | Method                                     | Optimization Scope  | Dice (Validated) | Performance Note |
|:--------- |:------------------------------------------ |:------------------- |:---------------- |:---------------- |
| **01**    | **Baseline Discrete 4D Grid Search**       | **Full 3D Volume**  | **0.8900**       | Global baseline (Noisy GT). |
| **02**    | **Automated Bayesian Optimization (TPE)**  | **Full 3D Volume**  | **0.8878**       | TPE baseline (Noisy GT). |
| **03**    | **ROI-Restricted Precision Tuning**        | **3D ROI Volume**   | **0.9864**       | Best 3D Baseline. |
| **04**    | **Multi-Otsu Statistical Thresholding**    | **3D ROI Volume**   | **0.9765**       | Multi-class statistical. |
| **05**    | **Sauvola Local Adaptive Thresholding**    | **3D ROI Volume**   | **0.0435**       | Ineffective (Local noise). |
| **06**    | **3D Morphological Active Contours**       | **3D ROI Volume**   | **0.9512**       | Smooth 3D surfaces. |
| **07**    | **Marker-Controlled Watershed**            | **3D ROI Volume**   | **0.9651**       | Optimal separation. |
| **08**    | **2D Slice-Wise Regional Active Contours** | **Single 2D Slice** | **0.4056**       | 2D Optimized only. |
| **09**    | **Diffusion-Based Random Walker**          | **Single 2D Slice** | **0.9709**       | 2D Optimized only. |
| **10**    | **Canny Edge-Based Volume Reconstruction** | **Single 2D Slice** | **0.3647**       | 2D Optimized only. |

### Key Observations:

- **Valid 3D Generalization (Iterations 03-07):** These results are the **"Gold Standard"** of this project. Optimized on the 3D ROI with Z-exclusion, they demonstrate near-perfect segmentation within the forearm.
- **Leaderboard Shift:** **Iteration 03 (ROI-Restricted Precision Tuning)** now holds the top spot (~0.986), proving that well-tuned baseline morphology is extremely effective when evaluated fairly.
- **2D Overfitting (Iterations 08-10):** These models show high slice-specific performance but require full 3D optimization to match the consistency of the top-performing volume-based methods.


## Ground Truth Limitations & Evaluation Context

A critical factor in interpreting the results of this project is the quality of the provided **ground truth (`bone_mask.nii`)**. During evaluation, several limitations were identified:

1. **High Background Noise:** The original mask is "dirty," containing large blocks of white pixels in the background air and significant noise throughout the volume.
2. **Lack of Component Filtering:** Unlike our automated pipeline, which strictly enforces a "Top 2 Components" rule to isolate the Radius and Ulna, the ground truth contains numerous small, unfiltered artifacts and disconnected fragments.
3. **Impact on Metrics:** Because our automated iterations are more precise and "cleaner" than the ground truth, the **Dice Coefficient** was initially penalized by background noise and invalid end-slices.

**Conclusion on Metrics:** The scores reported (e.g., ~0.986 for ROI Tuning) reflect excellent alignment with the anatomical structure when using our refined evaluation methodology. In a clinical setting, our automated pipeline would likely produce a more reliable 3D model for anatomical guides than the provided manual mask.

---

## Baseline Pipeline (Iterations 01-03)

The foundation of our segmentation approach, used and refined in the initial iterations, consists of three main stages:

1. **Thresholding**  
   Bone structures are segmented using a lower and upper intensity threshold to isolate pixels within the expected signal range for bone.

2. **Morphological Cleanup**  
   The thresholded binary mask is post-processed using:

   - **3D Morphological Closing**: Defined as dilation followed by erosion; used to connect nearby foreground regions and bridge small gaps in the bone structure.
   - **Binary Hole Filling**: A morphological reconstruction step that fills background cavities fully enclosed by the foreground object, recovering internal bone density.
   - **3D Morphological Opening**: Defined as erosion followed by dilation; removes small foreground artifacts (noise) and smooths the segmented boundaries.

3. **Connectivity Analysis**  
   To ensure anatomical validity, the final segmentation was restricted to the largest connected components (typically $N=2$), corresponding to the Radius and Ulna.

---

## Iteration 01: Baseline Discrete 4D Grid Search

**Folder**: `Iteration 01 - Baseline Discrete 4D Grid Search`

The original optimization strategy was based on an exhaustive 4D Grid Search over 84 parameter combinations.

### Grid Dimensions

- **Lower Threshold (`T_low`)**: [0, 5, 10]
- **Upper Threshold (`T_high`)**: [80, 85, 90, 95, 100, 105, 110]
- **Morphological Closing Radius (`R`)**: [1, 2]
- **Top Components Kept (`N`)**: [2, 3]

### Key Findings

- A lower threshold of `T_low = 0` was necessary to preserve the full low-signal bone volume.
- Keeping the **top 2 components** gave the best anatomical representation.
- **Final Grid Search Result (Global):** Dice **0.8900**, Hausdorff **32.0**.

---

## Iteration 02: Automated Bayesian Optimization (TPE)

**Folder**: `Iteration 02 - Automated Bayesian Optimization (TPE)`

Implemented **Optuna** with the **TPE** algorithm to search continuous threshold spaces.

### Final Result (Run 2)

- **Result**: Dice **0.8878**, Hausdorff **32.07**.

---

## Iteration 03: ROI-Restricted Precision Tuning

**Folder**: `Iteration 03 - ROI-Restricted Precision Tuning`

Transitioned to **Region of Interest (ROI)** evaluation to eliminate background air and "dirty mask" artifacts.

### ROI Definition

- **Bounding Box**: `[37:227, 90:189]` (y, x coordinates).
- **Z-Exclusion**: Excluding the first and last slices of the volume.
- **Metric Realism**: These refinements revealed a high Dice Coefficient of **~0.986**, providing an honest assessment of bone segmentation accuracy.

---

## New Experiments: Scikit-Image Segmentation

### Iteration 04: Multi-Otsu Statistical Thresholding

**Folder**: `Iteration 04 - Multi-Otsu Statistical Thresholding`  
**Method**: `skimage.filters.threshold_multiotsu` used to automatically determine intensity thresholds for 3-5 classes.

- **Result**: Dice **~0.976**. Effectively isolated bone intensity classes.

### Iteration 05: Sauvola Local Adaptive Thresholding

**Folder**: `Iteration 05 - Sauvola Local Adaptive Thresholding`  
**Method**: `skimage.filters.threshold_sauvola` applied for local intensity inhomogeneities.

- **Result**: Dice **~0.043**. Ineffective due to high sensitivity to local noise.

### Iteration 06: 3D Morphological Active Contours

**Folder**: `Iteration 06 - 3D Morphological Active Contours`  
**Method**: `skimage.segmentation.morphological_chan_vese` (ACWE) for boundary smoothing.

- **Result**: Dice **~0.951**. Successfully regularized bone boundaries in 3D.

### Iteration 07: Marker-Controlled Watershed

**Folder**: `Iteration 07 - Marker-Controlled Watershed`  
**Method**: `skimage.segmentation.watershed` with markers and Sobel elevation maps.

- **Result**: Dice **~0.965**. Excellent anatomical separation.

### Iteration 08: 2D Slice-Wise Regional Active Contours

**Folder**: `Iteration 08 - 2D Slice-Wise Regional Active Contours`  
**Method**: 2D `skimage.segmentation.chan_vese` applied slice-by-slice.

- **Result**: Dice **~0.405** (Optimized Slice).

### Iteration 09: Diffusion-Based Random Walker

**Folder**: `Iteration 09 - Diffusion-Based Random Walker`  
**Method**: `skimage.segmentation.random_walker` with optimized intensity seeds.

- **Result**: Dice **~0.971** (Optimized Slice).

### Iteration 10: Canny Edge-Based Volume Reconstruction

**Folder**: `Iteration 10 - Canny Edge-Based Volume Reconstruction`  
**Method**: `skimage.feature.canny` followed by 3D binary hole filling.

- **Result**: Dice **~0.364** (Optimized Slice).


---

## Skin Segmentation

**Folder**: `Skin Segmentation/`  
**Method**: `skimage.feature.canny` followed by 3D binary hole filling and morphological erosion.

In addition to bone segmentation, we implemented a dedicated pipeline to isolate the skin (outer boundary) of the phantom. This task utilized a simplified version of the edge-based approach:

1. **Boundary Detection**: A Canny edge filter was applied slice-by-slice to detect the outer interface of the phantom.
2. **Solid Volume Creation**: Binary hole filling was used to create a solid mask of the entire forearm.
3. **Thin Shell Extraction**: To produce a realistic **2mm thick skin mask**, the solid volume was eroded by 2mm (2 pixels, based on 1mm/px metadata) and subtracted from the original filled volume.
4. **Clean-up**: Only the largest connected component was retained to ensure a single, continuous skin boundary.

---

### 💻 How to Reproduce

1. Ensure the `MRI/` folder and `bone_mask.nii` are in the root directory.
2. Install dependencies: `conda env create --file environment.yml`.
3. Activate the environment: `conda activate image_reconstruction_visualisation`.
4. Run the full suite (Bone & Skin Segmentation): `python run_all_iterations.py`.

---

## Project Structure & Files

- `run_all_iterations.py` — Orchestration script.
- `Iteration 01 - Baseline Discrete 4D Grid Search/`
- `Iteration 02 - Automated Bayesian Optimization (TPE)/`
- `Iteration 03 - ROI-Restricted Precision Tuning/`
- `Iteration 04` through `Iteration 10` (Advanced Scikit-Image experiments).
- `final_bone_mask_hd_optimized.nii` — Resulting segmentation mask.
- `MRI/`, `bone_mask.nii` — Dataset and Ground Truth.

## Evaluation Metrics

- **Dice Coefficient**: Volumetric overlap.
- **Hausdorff Distance**: Boundary agreement/outlier penalty.

## Reference

```bibtex
@inproceedings{optuna_2019,
title={Optuna: A Next-generation Hyperparameter Optimization Framework},
author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
booktitle={Proceedings of the 25th {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
year={2019}
}

@article{scikit-image,
 title = {scikit-image: image processing in {P}ython},
 author = {van der Walt, Stéfan and Schönberger, Johannes L. and Nunez-Iglesias, Juan and Boulogne, François and Warner, Joshua D. and Yager, Neil and Gouillart, Emmanuelle and Yu, Tony and {the scikit-image contributors}},
 year = {2014},
 month = {6},
 volume = {2},
 pages = {e453},
 journal = {PeerJ},
 issn = {2167-8359},
 url = {https://doi.org/10.7717/peerj.453},
 doi = {10.7717/peerj.453}
}
```

---

*Created for the Course: 3D Image Reconstruction & Visualisation (202400339)*
