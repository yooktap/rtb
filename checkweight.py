import nibabel as nib

img = nib.load("../sampledata/DL/sub-047EPKL011005_ses-1_t1w.nii")
header = img.header
print(header)
