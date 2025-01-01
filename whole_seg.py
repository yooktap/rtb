import glob
import os
import nibabel as nib

# Numpy for numpy.arrays
import numpy as np

# Include ITK for DICOM reading.
import itk

# Include pydicom_seg for DICOM SEG objects
import pydicom
import pydicom_seg

# for downloading data from TCIA
from tcia_utils import nbia

# This is the most common import command for itkWidgets.
#   The view() function opens an interactive viewer for 2D and 3D
#   data in a variety of formats.
from itkwidgets import view

# imports for monai
import torch
import monai
from monai.data import decollate_batch
from monai.bundle import ConfigParser, download

from monai.config import print_config

#print_config()

image_path = "sampledata/DL/sub-047EPKL011005_ses-1_t1w.nii"
image_data = nib.load(image_path).get_fdata()

# Create a MONAI dataset
dataset = monai.data.Dataset(data=[{"image": image_path}], transform=None)

# Create a DataLoader
data_loader = monai.data.DataLoader(dataset, batch_size=1)

# Access the image in the DataLoader
for batch_data in data_loader:
    image = batch_data["image"]
    print(image)
