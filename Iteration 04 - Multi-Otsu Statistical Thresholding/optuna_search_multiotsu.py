import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import closing, opening, ball
from scipy.ndimage import binary_fill_holes, label
from skimage.metrics import hausdorff_distance
from skimage.filters import threshold_multiotsu


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
    # Parameters for Multi-Otsu
    n_classes = trial.suggest_int('n_classes', 3, 5)
    
    # Pre-calculate thresholds for this trial
    # We do this once per trial on the ROI
    y1, y2, x1, x2 = ROI
    roi_pixels = mri_volume[1:-1, y1:y2, x1:x2]
    
    try:
        thresholds = threshold_multiotsu(roi_pixels, classes=n_classes)
    except Exception:
        # Sometimes Multi-Otsu fails if there aren't enough unique values
        return 0.0

    # Which class(es) to pick? 
    # Let's say we pick a range of classes [c_start, c_end]
    c_start = trial.suggest_int('c_start', 0, n_classes - 1)
    c_end = trial.suggest_int('c_end', c_start, n_classes - 1)
    
    # Create mask based on selected classes
    # Classes are:
    # 0: p < t[0]
    # 1: t[0] <= p < t[1]
    # ...
    # n-1: p >= t[n-2]
    
    seg_roi = np.zeros(roi_pixels.shape, dtype=bool)
    for i in range(c_start, c_end + 1):
        if i == 0:
            seg_roi |= (roi_pixels < thresholds[0])
        elif i == n_classes - 1:
            seg_roi |= (roi_pixels >= thresholds[-1])
        else:
            seg_roi |= (roi_pixels >= thresholds[i-1]) & (roi_pixels < thresholds[i])

    # Post-processing
    radius_closing = trial.suggest_int('radius_closing', 1, 5)
    radius_opening = trial.suggest_int('radius_opening', 1, 2)
    
    seg_roi = closing(seg_roi, ball(radius_closing))
    seg_roi = binary_fill_holes(seg_roi)
    seg_roi = opening(seg_roi, ball(radius_opening))
    
    # Reconstruct full volume for component selection (though we only care about ROI)
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
study.optimize(objective, n_trials=400, n_jobs=-1) # Reduced trials for faster execution in this turn

results = [
    {
        'trial': t.number,
        'n_classes': t.params['n_classes'],
        'c_start': t.params['c_start'],
        'c_end': t.params['c_end'],
        'radius_closing': t.params['radius_closing'],
        'radius_opening': t.params['radius_opening'],
        'dice': t.user_attrs['dice'],
        'hausdorff': t.user_attrs['hd'],
        'score': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

df = pd.DataFrame(results).sort_values('score', ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_multiotsu.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_multiotsu.csv'), index=False)

latex_table = top_results.to_latex(
    index=False,
    caption='Top 10 configurations for Multi-Otsu Segmentation',
    label='tab:multiotsu_top',
    float_format='%.4f'
)
with open(os.path.join(script_dir, 'top_optuna_results_latex.txt'), 'w') as f:
    f.write(latex_table)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
