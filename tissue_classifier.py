from dipy.segment.tissue import TissueClassifierHMRF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

file_path = '../test/sub-047EPKL011005_ses-1_t1w.niiBrainExtractionBrain.nii'
img = nib.load(file_path)
t1 = img.get_fdata()

nclass = 3
beta = 0.1

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)

fig = plt.figure()
a = fig.add_subplot(1, 4, 1)
img_ax = np.rot90(PVE[:, :, 100, 0])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('CSF')
a = fig.add_subplot(1, 4, 2)
img_cor = np.rot90(PVE[:, :, 100, 1])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Gray Matter')
a = fig.add_subplot(1, 4, 3)
img_cor = np.rot90(PVE[:, :, 100, 2])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('White Matter')
a = fig.add_subplot(1, 4, 4)
img_cor = np.rot90(t1[:, :, 100])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('T1')
plt.show()