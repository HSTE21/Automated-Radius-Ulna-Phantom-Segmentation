import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import closing, ball, opening
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from tqdm import tqdm

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

def evaluate_params(args):
    low, high, radius, top_n, mri_volume, ground_truth = args
    
    # Segmentation
    seg = (mri_volume >= low) & (mri_volume <= high)
    seg = closing(seg, ball(radius))
    seg = binary_fill_holes(seg)
    seg = opening(seg, ball(1))
    seg_final = keep_top_n_components(seg, top_n)
    
    dice = dice_coefficient(ground_truth[1:-1], seg_final[1:-1])
    hd = hausdorff_distance(ground_truth[1:-1], seg_final[1:-1])
    
    return {
        'T_low': low,
        'T_high': high,
        'Radius': radius,
        'Top_N': top_n,
        'Dice': dice,
        'Hausdorff': hd
    }

def main():
    print("Loading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    mri_path = os.path.join(root_dir, 'MRI')
    gt_path = os.path.join(root_dir, 'bone_mask.nii')
    
    mri_volume = load_dicom_series(mri_path).astype(float)
    ground_truth = nib.load(gt_path).get_fdata().astype(bool)
    
    # Grid dimensions from README/REPORT
    low_thresholds = [0, 5, 10]
    high_thresholds = [80, 85, 90, 95, 100, 105, 110]
    radii = [1, 2]
    top_ns = [2, 3]
    
    tasks = []
    for low in low_thresholds:
        for high in high_thresholds:
            for r in radii:
                for n in top_ns:
                    tasks.append((low, high, r, n, mri_volume, ground_truth))
    
    print(f"Starting Grid Search (84 combinations)...")
    results = []
    for task in tqdm(tasks):
        results.append(evaluate_params(task))
    
    df = pd.DataFrame(results).sort_values(by='Dice', ascending=False)
    
    # 1. Save as CSV file (full results)
    df.to_csv(os.path.join(script_dir, 'optuna_results_grid.csv'), index=False)
    
    # 2. Save top results as separate CSV
    top_results = df.head(10)
    top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_grid.csv'), index=False)
    
    # 3. Save top results as LaTeX format in a .txt file
    latex_table = top_results.to_latex(index=False, 
                                     caption="Top 10 Parameter Configurations from 84-point Grid Search",
                                     label="tab:grid_search_top",
                                     float_format="%.4f")
    
    with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
        f.write(latex_table)
    
    print("\nTop 5 Results:")
    print(top_results.head(5))
    print("\nGrid search complete.")
    print(f"- Full results saved to: {os.path.join(script_dir, 'optuna_results_grid.csv')}")
    print(f"- Top results (LaTeX) saved to: {os.path.join(script_dir, 'top_optuna_results_latex.txt')}")

if __name__ == "__main__":
    main()
