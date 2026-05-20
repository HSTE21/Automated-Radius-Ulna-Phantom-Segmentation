import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import ball, closing
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.feature import canny


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
    # Parameters for Canny
    sigma = trial.suggest_float('sigma', 0.1, 5.0)
    low_threshold = trial.suggest_float('low_threshold', 0, 50)
    high_threshold = trial.suggest_float('high_threshold', low_threshold, 150)
    
    y1, y2, x1, x2 = ROI
    slice_idx = mri_volume.shape[0] // 2
    roi_slice = mri_volume[slice_idx, y1:y2, x1:x2]
    
    # Normalize slice to [0, 1] for Canny
    roi_norm = (roi_slice - np.min(roi_slice)) / (np.max(roi_slice) - np.min(roi_slice) + 1e-8)
    
    try:
        edges = canny(roi_norm, sigma=sigma, low_threshold=low_threshold/255.0, high_threshold=high_threshold/255.0)
        # Morphological fill to get object from edges
        filled = binary_fill_holes(edges)
    except Exception:
        return 0.0

    dice = dice_coefficient(ground_truth[slice_idx, y1:y2, x1:x2], filled)
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
        'sigma': t.params['sigma'],
        'low_threshold': t.params['low_threshold'],
        'high_threshold': t.params['high_threshold'],
        'dice_slice': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]


df = pd.DataFrame(results)
df = df.sort_values(df.columns[-1], ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_canny.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_canny.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 parameter configurations for canny',
    label='tab:canny_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
