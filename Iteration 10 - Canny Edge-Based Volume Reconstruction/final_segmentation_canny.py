import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import ball, closing
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.feature import canny
from tqdm import tqdm

def load_dicom_series(directory):
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([f.pixel_array for f in files]).astype(float)

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_total = np.sum(y_true) + np.sum(y_pred)
    if sum_total == 0: return 1.0
    return (2. * intersection) / sum_total

def keep_top_2_components(mask):
    labeled, n = label(mask)
    if n <= 2:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    top2 = np.argsort(sizes)[-2:]
    return np.isin(labeled, top2)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    mri_path = os.path.join(root_dir, 'MRI')
    gt_path = os.path.join(root_dir, 'bone_mask.nii')
    
    print("Loading data...")
    mri_volume = load_dicom_series(mri_path).astype(float)
    ground_truth = nib.load(gt_path).get_fdata().astype(bool)
    
    ROI = (37, 227, 90, 189)
    y1, y2, x1, x2 = ROI
    
    results_file = os.path.join(script_dir, 'optuna_results_canny.csv')
    df = pd.read_csv(results_file).sort_values(by='dice_slice', ascending=False)
    best_params = df.iloc[0]
    
    sigma = best_params['sigma']
    low_threshold = best_params['low_threshold']
    high_threshold = best_params['high_threshold']
    
    print(f"Applying Canny Edge (slice by slice) with sigma={sigma}, low_threshold={low_threshold}, high_threshold={high_threshold}")
    
    roi_volume = mri_volume[1:-1, y1:y2, x1:x2]
    seg_roi = np.zeros(roi_volume.shape, dtype=bool)
    
    for i in tqdm(range(roi_volume.shape[0])):
        roi_slice = roi_volume[i]
        # Normalize slice to [0, 1] for Canny
        roi_norm = (roi_slice - np.min(roi_slice)) / (np.max(roi_slice) - np.min(roi_slice) + 1e-8)
        
        try:
            edges = canny(roi_norm, sigma=sigma, low_threshold=low_threshold/255.0, high_threshold=high_threshold/255.0)
            # Morphological fill to get object from edges
            seg_roi[i] = binary_fill_holes(edges)
        except Exception:
            pass
            
    # Post-processing
    seg_roi = closing(seg_roi, ball(1))
    
    # Reconstruct full volume
    seg = np.zeros_like(mri_volume, dtype=bool)
    seg[1:-1, y1:y2, x1:x2] = seg_roi
    
    # Component selection
    seg = keep_top_2_components(seg)

    dice = dice_coefficient(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    hd = hausdorff_distance(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    
    print(f"Final Dice: {dice:.4f}")
    print(f"Final Hausdorff: {hd:.4f}")
    
    slice_idx = mri_volume.shape[0] // 2
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(mri_volume[slice_idx], cmap='gray')
    plt.title('Original MRI')
    plt.subplot(132)
    plt.imshow(ground_truth[slice_idx], cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(133)
    plt.imshow(seg[slice_idx], cmap='gray')
    plt.title(f'Canny Edge (Dice={dice:.4f})')
    plt.savefig(os.path.join(script_dir, 'final_segmentation_canny.png'))
    
    nib.save(nib.Nifti1Image(seg.astype(np.float32), np.eye(4)), 
             os.path.join(script_dir, 'final_bone_mask_canny.nii'))
    print(f"Results saved in {script_dir}")

if __name__ == "__main__":
    main()
