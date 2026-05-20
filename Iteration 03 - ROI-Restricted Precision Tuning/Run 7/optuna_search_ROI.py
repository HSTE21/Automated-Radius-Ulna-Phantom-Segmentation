import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import closing, opening, ball
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance


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
    t_low = trial.suggest_float('t_low', 0, 150)
    t_high = trial.suggest_float('t_high', t_low, 150)
    radius = trial.suggest_int('radius', 1, 5)

    seg = np.zeros_like(mri_volume, dtype=bool)
    y1, y2, x1, x2 = ROI
    roi_pixels = mri_volume[1:-1, y1:y2, x1:x2]
    seg[1:-1, y1:y2, x1:x2] = (roi_pixels >= t_low) & (roi_pixels <= t_high)

    # Post-processing ALLEEN op ROI
    seg_roi = seg[1:-1, y1:y2, x1:x2].copy()
    seg_roi = closing(seg_roi, ball(radius))
    seg_roi = binary_fill_holes(seg_roi)
    seg_roi = opening(seg_roi, ball(1))
    
    # Terug in volume
    seg[1:-1, y1:y2, x1:x2] = seg_roi
    
    # Component selection op hele volume
    seg = keep_top_2_components(seg)

    dice = dice_coefficient(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    hd = hausdorff_distance(ground_truth[1:-1, y1:y2, x1:x2], seg[1:-1, y1:y2, x1:x2])
    
    trial.set_user_attr("dice", float(dice))
    trial.set_user_attr("hd", float(hd))

    return dice - 0.001 * hd


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
mri_path = os.path.join(root_dir, 'MRI')
gt_path = os.path.join(root_dir, 'bone_mask.nii')

mri_volume = load_dicom_series(mri_path)
ground_truth = nib.load(gt_path).get_fdata().astype(bool)
ROI = (37, 227, 90, 189)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=800, n_jobs=-1)

results = [
    {
        'trial': t.number,
        't_low': t.params['t_low'],
        't_high': t.params['t_high'],
        'radius': t.params['radius'],
        'dice': t.user_attrs['dice'],
        'hausdorff': t.user_attrs['hd'],
        'score': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

df = pd.DataFrame(results).sort_values('score', ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_roi.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_roi.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 parameterconfiguraties uit Optuna-optimalisatie',
    label='tab:optuna_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, os.path.join(script_dir, 'top_optuna_results_latex.txt')), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')