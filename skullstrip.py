import subprocess

# **Step 1: Skull Stripping**
print("Performing skull stripping...")

subprocess.run (["bash", "antsBrainExtraction.sh", "-d 3", "-a /Users/yookta/ACSEF25/dyslexia/dyslexia/sampledata/DL/sub-047EPKL011005_ses-1_t1w.nii", "-e sampledata/Template/MNI152_T1_1mm-2.nii", "-m sampledata/Template/MNI152_T1_1mm_Brain_Mask.nii", "-o procdata/DL_stripped/anat_Stripped.nii"])