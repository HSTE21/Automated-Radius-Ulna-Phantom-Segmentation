import os
import optuna
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import ball, disk, closing
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
from skimage.metrics import hausdorff_distance
from skimage.segmentation import watershed
from skimage.filters import sobel


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
    # Parameters for Watershed
    t_marker_bg = trial.suggest_float('t_marker_bg', 100, 255)
    t_marker_fg = trial.suggest_float('t_marker_fg', 0, 50)
    
    y1, y2, x1, x2 = ROI
    roi_pixels = mri_volume[1:-1, y1:y2, x1:x2]
    
    # Elevation map
    elevation_map = sobel(roi_pixels)
    
    # Markers
    markers = np.zeros(roi_pixels.shape, dtype=int)
    markers[roi_pixels > t_marker_bg] = 1 # Background marker
    markers[roi_pixels < t_marker_fg] = 2 # Foreground (bone) marker
    
    # Run Watershed
    try:
        seg_roi_labeled = watershed(elevation_map, markers)
        seg_roi = (seg_roi_labeled == 2)
    except Exception:
        return 0.0

    # Post-processing
    radius_closing = trial.suggest_int('radius_closing', 1, 3)
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
# Using n_jobs=-1 for multi-CPU processing
study.optimize(objective, n_trials=400, n_jobs=-1)

results = [
    {
        'trial': t.number,
        't_marker_bg': t.params['t_marker_bg'],
        't_marker_fg': t.params['t_marker_fg'],
        'radius_closing': t.params['radius_closing'],
        'dice': t.user_attrs['dice'],
        'hausdorff': t.user_attrs['hd'],
        'score': t.value
    }
    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

df = pd.DataFrame(results).sort_values('score', ascending=False)
df.to_csv(os.path.join(script_dir, 'optuna_results_watershed.csv'), index=False)

top_results = df.head(10)
top_results.to_csv(os.path.join(script_dir, 'top_optuna_results_watershed.csv'), index=False)

print(f'Beste score: {study.best_value:.4f}')
print(f'Beste parameters: {study.best_params}')
