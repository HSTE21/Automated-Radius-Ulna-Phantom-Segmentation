import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import closing, opening, ball
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.segmentation import morphological_chan_vese


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
    # Parameters for Morphological Chan-Vese
    num_iter = trial.suggest_int('num_iter', 5, 30)
    smoothing = trial.suggest_int('smoothing', 1, 4)
    lambda1 = trial.suggest_int('lambda1', 1, 5)
    lambda2 = trial.suggest_int('lambda2', 1, 5)
    
    # Initialization threshold
    t_init = trial.suggest_float('t_init', 20, 100)
    
    # Post-processing (suggest here to avoid KeyError if return early)
    radius_closing = trial.suggest_int('radius_closing', 1, 3)
    
    y1, y2, x1, x2 = ROI
    roi_pixels = mri_volume[1:-1, y1:y2, x1:x2]
    
    # Initial mask
    init_mask = roi_pixels < t_init
    
    # Run Morph ACWE
    try:
        seg_roi = morphological_chan_vese(
            roi_pixels, 
            num_iter=num_iter, 
            init_level_set=init_mask, 
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2
        )
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

    # Post-processing
    seg_roi = closing(seg_roi, ball(radius_closing))
    seg_roi = binary_fill_holes(seg_roi)
    
    # Reconstruct full volume
    seg = np.zeros_like(mri_volume, dtype=bool)
    seg[1:-1, y1:y2, x1:x2] = seg_roi
    
    # Component selection
    seg = keep_top_2_components(seg)

    dice = dice_coefficient(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    hd = hausdorff_distance(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    
    trial.set_user_attr("dice", float(dice))
    trial.set_user_attr("hd", float(hd))

    return dice - 0.001 * hd


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
mri_path = os.path.join(root_dir, 'MRI')
gt_path = os.path.join(root_dir, 'bone_mask.nii')

mri_volume = load_dicom_series(mri_path)
ground_truth = nib.load(gt_path).get_fdata().astype(bool)
ROI = (37, 227, 90, 189)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=400, n_jobs=-1) # Morph ACWE is slow

results = [
    {
        'trial': t.number,
        'num_iter': t.params['num_iter'],
        'smoothing': t.params['smoothing'],
        'lambda1': t.params['lambda1'],
        'lambda2': t.params['lambda2'],
        't_init': t.params['t_init'],
        'radius_closing': t.params['radius_closing'],
        'dice': t.user_attrs['dice'],
        'hausdorff': t.user_attrs['hd'],
        'score': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

df = pd.DataFrame(results).sort_values('score', ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_cv.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_cv.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 parameter configurations for Morphological Chan-Vese',
    label='tab:cv_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
