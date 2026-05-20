import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import ball, closing
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.segmentation import chan_vese


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
    # Parameters for Standard Chan-Vese
    # Note: Standard Chan-Vese is 2D in scikit-image, we apply slice by slice or use a representative slice for speed.
    # To optimize for 3D, we'll pick a few representative slices or just optimize 2D parameters.
    
    mu = trial.suggest_float('mu', 0.0, 1.0)
    lambda1 = trial.suggest_float('lambda1', 0.5, 2.0)
    lambda2 = trial.suggest_float('lambda2', 0.5, 2.0)
    tol = trial.suggest_float('tol', 1e-4, 1e-2, log=True)
    max_num_iter = trial.suggest_int('max_num_iter', 10, 100)
    
    y1, y2, x1, x2 = ROI
    
    # We take the middle slice for fast optimization
    slice_idx = mri_volume.shape[0] // 2
    roi_slice = mri_volume[slice_idx, y1:y2, x1:x2]
    
    try:
        # standard chan_vese is 2D
        cv_seg = chan_vese(roi_slice, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tol, max_num_iter=max_num_iter)
    except Exception:
        return 0.0

    # Evaluate on this slice
    dice = dice_coefficient(ground_truth[slice_idx, y1:y2, x1:x2], cv_seg)
    
    # We return Dice as objective for parameter search
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
        'mu': t.params['mu'],
        'lambda1': t.params['lambda1'],
        'lambda2': t.params['lambda2'],
        'tol': t.params['tol'],
        'max_num_iter': t.params['max_num_iter'],
        'dice_slice': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]


df = pd.DataFrame(results)
df = df.sort_values(df.columns[-1], ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_cv_standard.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_cv_standard.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 parameter configurations for cv_standard',
    label='tab:cv_standard_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
