import nibabel as nib
import numpy as np
import sys
import matplotlib.pyplot as plt

# Load the NIfTI file
file_path = '/Users/yookip/ACSEF/wholeBrainSeg_Large_UNEST_segmentation/dataset/images/sub-047EPKL011005_ses-1_t1w.nii.gz'  # Replace with your file path
img = nib.load(file_path)
data = img.get_fdata()

# Get the middle slice in each dimension (axial, sagittal, coronal)
slice_0 = data[data.shape[0] // 2, :, :]  # Sagittal slice
slice_1 = data[:, data.shape[1] // 2, :]  # Coronal slice
slice_2 = data[:, :, data.shape[2] // 2]  # Axial slice

# Plot the slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(slice_0.T, cmap='terrain', origin='lower')
axes[0].set_title('Sagittal View')
axes[1].imshow(slice_1.T, cmap='terrain', origin='lower')
axes[1].set_title('Coronal View')
axes[2].imshow(slice_2.T, cmap='terrain', origin='lower')
axes[2].set_title('Axial View')

# Remove axes for a cleaner look
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
