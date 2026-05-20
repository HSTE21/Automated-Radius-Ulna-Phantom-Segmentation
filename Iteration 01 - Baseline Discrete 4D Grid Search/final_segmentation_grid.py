import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import closing, ball, opening
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance

def load_dicom_series(directory):
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(x.InstanceNumber))
    pixel_data = np.stack([f.pixel_array for f in files])
    return pixel_data

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_total = np.sum(y_true) + np.sum(y_pred)
    if sum_total == 0: return 1.0
    return (2. * intersection) / sum_total

def keep_top_n_components(mask, n):
    labeled, num_features = label(mask)
    if num_features <= n: return mask
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    top_n_indices = np.argsort(component_sizes)[::-1][:n]
    new_mask = np.zeros_like(mask, dtype=bool)
    for idx in top_n_indices:
        if component_sizes[idx] > 0:
            new_mask |= (labeled == idx)
    return new_mask

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    mri_path = os.path.join(root_dir, 'MRI')
    gt_path = os.path.join(root_dir, 'bone_mask.nii')
    
    print("Loading data...")
    mri_volume = load_dicom_series(mri_path).astype(float)
    ground_truth = nib.load(gt_path).get_fdata().astype(bool)
    
    results_file = os.path.join(script_dir, 'optuna_results_grid.csv')
    df = pd.read_csv(results_file).sort_values(by='Dice', ascending=False)
    best_params = df.iloc[0]
    
    low = best_params['T_low']
    high = best_params['T_high']
    radius = int(best_params['Radius'])
    top_n = int(best_params['Top_N'])
    
    print(f"Applying segmentation with T_low={low}, T_high={high}, Radius={radius}, Top_N={top_n}")
    
    # Segmentation
    seg = (mri_volume >= low) & (mri_volume <= high)
    seg = closing(seg, ball(radius))
    seg = binary_fill_holes(seg)
    seg = opening(seg, ball(1))
    seg_final = keep_top_n_components(seg, top_n)
    
    dice = dice_coefficient(ground_truth[1:-1], seg_final[1:-1])
    hd = hausdorff_distance(ground_truth[1:-1], seg_final[1:-1])
    
    print(f"Final Dice: {dice:.4f}")
    print(f"Final Hausdorff: {hd:.4f}")
    
    # Save results
    nib.save(nib.Nifti1Image(seg_final.astype(np.float32), np.eye(4)), 
             os.path.join(script_dir, 'final_bone_mask_grid_optimized.nii'))
    
    slice_idx = mri_volume.shape[0] // 2
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(mri_volume[slice_idx], cmap='gray')
    plt.title('Original MRI')
    plt.subplot(132)
    plt.imshow(ground_truth[slice_idx], cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(133)
    plt.imshow(seg_final[slice_idx], cmap='gray')
    plt.title(f'Segmentation (Dice={dice:.4f})')
    plt.savefig(os.path.join(script_dir, 'final_segmentation_grid.png'))
    print(f"Results saved in {script_dir}")

if __name__ == "__main__":
    main()
