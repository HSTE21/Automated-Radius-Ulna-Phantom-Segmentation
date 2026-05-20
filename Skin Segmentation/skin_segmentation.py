import os
import pydicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage.feature import canny
from tqdm import tqdm

def load_dicom_series(directory):
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([f.pixel_array for f in files]).astype(float)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    mri_path = os.path.join(root_dir, 'MRI')
    
    print("Loading DICOM data...")
    mri_volume = load_dicom_series(mri_path)
    
    print("Extracting 2mm skin boundary...")
    # Based on metadata, 1 pixel = 1mm. So 2mm = 2 pixels.
    thickness_px = 2
    
    full_filled_mask = np.zeros(mri_volume.shape, dtype=bool)
    
    # First pass: Generate filled volumes for each slice
    for i in tqdm(range(mri_volume.shape[0]), desc="Filling volumes"):
        slice_data = mri_volume[i]
        mi, ma = np.min(slice_data), np.max(slice_data)
        if ma - mi > 0:
            slice_norm = (slice_data - mi) / (ma - mi)
        else:
            slice_norm = np.zeros_like(slice_data)
            
        # Use Canny to find the outer boundary
        edges = canny(slice_norm, sigma=2.0)
        full_filled_mask[i] = binary_fill_holes(edges)
    
    # Second pass: Create the 2mm shell
    # We erode the filled volume and subtract it from the original to get the shell
    print(f"Creating {thickness_px}mm shell via erosion...")
    eroded_mask = binary_erosion(full_filled_mask, iterations=thickness_px)
    skin_mask = full_filled_mask ^ eroded_mask
    
    # Save the resulting NIfTI image
    output_nifti = os.path.join(script_dir, 'skin_mask.nii')
    nib.save(nib.Nifti1Image(skin_mask.astype(np.float32), np.eye(4)), output_nifti)
    
    # Visualization
    slice_idx = mri_volume.shape[0] // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mri_volume[slice_idx], cmap='gray')
    plt.title('Original MRI')
    plt.subplot(1, 2, 2)
    plt.imshow(skin_mask[slice_idx], cmap='gray')
    plt.title(f'Skin Mask ({thickness_px}mm thickness)')
    
    plt.savefig(os.path.join(script_dir, 'skin_segmentation.png'))
    print(f"Results saved in {script_dir}")

if __name__ == "__main__":
    main()
