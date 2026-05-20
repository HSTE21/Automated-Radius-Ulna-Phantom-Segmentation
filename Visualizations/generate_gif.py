import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import imageio
import os
import pydicom

def load_dicom_volume(directory):
    # Adjust directory path because script is now inside Visualizations/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mri_dir = os.path.join(base_dir, directory)
    
    files = [pydicom.dcmread(os.path.join(mri_dir, f)) for f in os.listdir(mri_dir) if f.endswith('.dcm')]
    # Use InstanceNumber to sort, matching the original project scripts
    files.sort(key=lambda x: int(x.InstanceNumber))
    # Stacking as (z, y, x) to match project conventions
    volume = np.stack([f.pixel_array for f in files], axis=0)
    return volume

if __name__ == "__main__":
    # Get base directory (root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "Visualizations")
    
    # Load MRI (z, y, x)
    print("Loading MRI DICOMs...")
    mri_vol = load_dicom_volume("MRI")
    
    # Use the best mask
    best_mask_path = os.path.join(base_dir, "Iteration 03 - ROI-Restricted Precision Tuning/Run 6/final_bone_mask_optuna_optimized.nii")
    print(f"Loading mask from {best_mask_path}...")
    mask_img = nib.load(best_mask_path)
    mask_data = mask_img.get_fdata() # This is likely (z, y, x) or (x, y, z)
    
    # In the project's scripts, NIfTI was saved from (z, y, x) volume
    # Let's ensure alignment. If mask_data was saved as (z, y, x), it stays (z, y, x).
    
    # Normalization of MRI
    mri_vol = (mri_vol - np.min(mri_vol)) / (np.max(mri_vol) - np.min(mri_vol))
    
    output_gif = os.path.join(output_dir, "segmentation_scroller.gif")
    frames = []
    
    # Correct orientation check: 
    # Usually pydicom pixel_array is (rows, cols) -> (y, x).
    # Matplotlib imshow(slice) displays y as vertical, x as horizontal.
    
    print("Generating frames (Corrected Orientation)...")
    # Loop over Z (the first dimension now)
    for z in range(1, mri_vol.shape[0] - 1):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # MRI Slice (y, x)
        mri_slice = mri_vol[z, :, :]
        ax.imshow(mri_slice, cmap='gray')
        
        # Mask Slice
        mask_slice = mask_data[z, :, :]
        mask_overlay = np.zeros((*mask_slice.shape, 4))
        mask_overlay[mask_slice > 0] = [0, 1, 0, 0.6] # Semi-transparent green
        
        ax.imshow(mask_overlay)
        
        ax.set_title(f"3D Bone Segmentation - Z-Slice: {z}")
        ax.axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image[:, :, :3]) 
        plt.close(fig)
        
    print(f"Saving GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, duration=0.1)
    print("Done!")
