import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import ball, closing
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.segmentation import random_walker


def load_dicom_series(directory):
    files = [
        pydicom.dcmread(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f.endswith('.dcm')
    ]
    files.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([f.pixel_array for f in files]).astype(float)


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    return 1.0 if total == 0 else (2.0 * intersection) / total


def keep_top_2_components(mask):
    labeled, n = label(mask)
    if n <= 2:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    top2 = np.argsort(sizes)[-2:]
    return np.isin(labeled, top2)


def objective(trial):
    # Parameters for Random Walker
    beta = trial.suggest_float('beta', 10, 1000, log=True)
    t_marker_bg = trial.suggest_float('t_marker_bg', 120, 255)
    t_marker_fg = trial.suggest_float('t_marker_fg', 0, 40)
    
    y1, y2, x1, x2 = ROI
    # Use 2D slices for speed in optimization, then evaluate in 3D
    slice_idx = mri_volume.shape[0] // 2
    roi_slice = mri_volume[slice_idx, y1:y2, x1:x2]
    
    markers = np.zeros(roi_slice.shape, dtype=int)
    markers[roi_slice > t_marker_bg] = 1 # Background
    markers[roi_slice < t_marker_fg] = 2 # Foreground
    
    # Random walker needs at least one seed of each label
    if not (np.any(markers == 1) and np.any(markers == 2)):
        return 0.0
    
    try:
        # random_walker is 2D/3D but slow in 3D
        rw_seg = random_walker(roi_slice, markers, beta=beta, mode='bf')
        seg_slice = (rw_seg == 2)
    except Exception:
        return 0.0

    dice = dice_coefficient(ground_truth[slice_idx, y1:y2, x1:x2], seg_slice)
    return dice


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
mri_path = os.path.join(root_dir, 'MRI')
gt_path = os.path.join(root_dir, 'bone_mask.nii')

mri_volume = load_dicom_series(mri_path)
ground_truth = nib.load(gt_path).get_fdata().astype(bool)
ROI = (37, 227, 90, 189)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=400, n_jobs=-1)

results = [
    {
        'trial': t.number,
        'beta': t.params['beta'],
        't_marker_bg': t.params['t_marker_bg'],
        't_marker_fg': t.params['t_marker_fg'],
        'dice_slice': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]


df = pd.DataFrame(results)
df = df.sort_values(df.columns[-1], ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_rw.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_rw.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 parameter configurations for rw',
    label='tab:rw_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
