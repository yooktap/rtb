from dipy.segment.tissue import TissueClassifierHMRF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob

segment_names = {0: "background", 1: "3rd-Ventricle", 2: "4th-Ventricle", 3: "Right-Accumbens-Area", 4: "Left-Accumbens-Area", 5: "Right-Amygdala",
    6: "Left-Amygdala", 7: "Brain-Stem", 8: "Right-Caudate", 9: "Left-Caudate", 10: "Right-Cerebellum-Exterior", 11: "Left-Cerebellum-Exterior",
    12: "Right-Cerebellum-White-Matter", 13: "Left-Cerebellum-White-Matter", 14: "Right-Cerebral-White-Matter", 15: "Left-Cerebral-White-Matter",
    16: "Right-Hippocampus", 17: "Left-Hippocampus", 18: "Right-Inf-Lat-Vent", 19: "Left-Inf-Lat-Vent",
    20: "Right-Lateral-Ventricle", 21: "Left-Lateral-Ventricle", 22: "Right-Pallidum", 23: "Left-Pallidum",
    24: "Right-Putamen", 25: "Left-Putamen", 26: "Right-Thalamus-Proper", 27: "Left-Thalamus-Proper",
    28: "Right-Ventral-DC", 29: "Left-Ventral-DC", 30: "Cerebellar-Vermal-Lobules-I-V", 31: "Cerebellar-Vermal-Lobules-VI-VII",
    32: "Cerebellar-Vermal-Lobules-VIII-X", 33: "Left-Basal-Forebrain", 34: "Right-Basal-Forebrain", 35: "Right-ACgG--anterior-cingulate-gyrus",
    36: "Left-ACgG--anterior-cingulate-gyrus", 37: "Right-AIns--anterior-insula", 38: "Left-AIns--anterior-insula", 39: "Right-AOrG--anterior-orbital-gyrus",
    40: "Left-AOrG--anterior-orbital-gyrus", 41: "Right-AnG---angular-gyrus", 42: "Left-AnG---angular-gyrus", 43: "Right-Calc--calcarine-cortex",
    44: "Left-Calc--calcarine-cortex", 45: "Right-CO----central-operculum", 46: "Left-CO----central-operculum", 47: "Right-Cun---cuneus",
    48: "Left-Cun---cuneus", 49: "Right-Ent---entorhinal-area", 50: "Left-Ent---entorhinal-area", 51: "Right-FO----frontal-operculum",
    52: "Left-FO----frontal-operculum", 53: "Right-FRP---frontal-pole", 54: "Left-FRP---frontal-pole", 55: "Right-FuG---fusiform-gyrus",
    56: "Left-FuG---fusiform-gyrus", 57: "Right-GRe---gyrus-rectus", 58: "Left-GRe---gyrus-rectus", 59: "Right-IOG---inferior-occipital-gyrus",
    60: "Left-IOG---inferior-occipital-gyrus", 61: "Right-ITG---inferior-temporal-gyrus", 62: "Left-ITG---inferior-temporal-gyrus", 63: "Right-LiG---lingual-gyrus",
    64: "Left-LiG---lingual-gyrus", 65: "Right-LOrG--lateral-orbital-gyrus", 66: "Left-LOrG--lateral-orbital-gyrus", 67: "Right-MCgG--middle-cingulate-gyrus",
    68: "Left-MCgG--middle-cingulate-gyrus", 69: "Right-MFC---medial-frontal-cortex", 70: "Left-MFC---medial-frontal-cortex", 71: "Right-MFG---middle-frontal-gyrus",
    72: "Left-MFG---middle-frontal-gyrus", 73: "Right-MOG---middle-occipital-gyrus", 74: "Left-MOG---middle-occipital-gyrus", 75: "Right-MOrG--medial-orbital-gyrus",
    76: "Left-MOrG--medial-orbital-gyrus", 77: "Right-MPoG--postcentral-gyrus", 78: "Left-MPoG--postcentral-gyrus", 79: "Right-MPrG--precentral-gyrus",
    80: "Left-MPrG--precentral-gyrus", 81: "Right-MSFG--superior-frontal-gyrus", 82: "Left-MSFG--superior-frontal-gyrus", 83: "Right-MTG---middle-temporal-gyrus",
    84: "Left-MTG---middle-temporal-gyrus", 85: "Right-OCP---occipital-pole", 86: "Left-OCP---occipital-pole", 87: "Right-OFuG--occipital-fusiform-gyrus",
    88: "Left-OFuG--occipital-fusiform-gyrus", 89: "Right-OpIFG-opercular-part-of-the-IFG", 90: "Left-OpIFG-opercular-part-of-the-IFG", 91: "Right-OrIFG-orbital-part-of-the-IFG",
    92: "Left-OrIFG-orbital-part-of-the-IFG", 93: "Right-PCgG--posterior-cingulate-gyrus", 94: "Left-PCgG--posterior-cingulate-gyrus", 95: "Right-PCu---precuneus",
    96: "Left-PCu---precuneus", 97: "Right-PHG---parahippocampal-gyrus", 98: "Left-PHG---parahippocampal-gyrus", 99: "Right-PIns--posterior-insula",
    100: "Left-PIns--posterior-insula", 101: "Right-PO----parietal-operculum", 102: "Left-PO----parietal-operculum", 103: "Right-PoG---postcentral-gyrus",
    104: "Left-PoG---postcentral-gyrus", 105: "Right-POrG--posterior-orbital-gyrus", 106: "Left-POrG--posterior-orbital-gyrus", 107: "Right-PP----planum-polare",
    108: "Left-PP----planum-polare", 109: "Right-PrG---precentral-gyrus", 110: "Left-PrG---precentral-gyrus", 111: "Right-PT----planum-temporale",
    112: "Left-PT----planum-temporale", 113: "Right-SCA---subcallosal-area", 114: "Left-SCA---subcallosal-area", 115: "Right-SFG---superior-frontal-gyrus",
    116: "Left-SFG---superior-frontal-gyrus", 117: "Right-SMC---supplementary-motor-cortex", 118: "Left-SMC---supplementary-motor-cortex", 119: "Right-SMG---supramarginal-gyrus",
    120: "Left-SMG---supramarginal-gyrus", 121: "Right-SOG---superior-occipital-gyrus", 122: "Left-SOG---superior-occipital-gyrus", 123: "Right-SPL---superior-parietal-lobule",
    124: "Left-SPL---superior-parietal-lobule", 125: "Right-STG---superior-temporal-gyrus", 126: "Left-STG---superior-temporal-gyrus", 127: "Right-TMP---temporal-pole",
    128: "Left-TMP---temporal-pole", 129: "Right-TrIFG-triangular-part-of-the-IFG", 130: "Left-TrIFG-triangular-part-of-the-IFG", 131: "Right-TTG---transverse-temporal-gyrus",
    132: "Left-TTG---transverse-temporal-gyrus"
}

class niiimg:
    def __init__(self, img_path=None, data=None):
        if img_path is not None:
            self.img_path = img_path
            self.img = nib.load(img_path)
            self.img_data = self.img.get_fdata()

        elif data is not None:
            self.img_data = data
            self.img_path = ''
        
        else:
            print("Please provide an image path or data.")
            return

        self.img_shape = self.img_data.shape
        self.PVE = None
        self.csf_mask = None
        self.gray_mask = None
        self.white_mask = None

    def plot_slices(self):
        # Get the middle slice in each dimension (axial, sagittal, coronal)
        slice_0 = self.img_data[self.img_shape[0] // 2, :, :]  # Sagittal slice
        slice_1 = self.img_data[:, self.img_shape[1] // 2, :]  # Coronal slice
        slice_2 = self.img_data[:, :, self.img_shape[2] // 2]  # Axial slice

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

    def save_nii(self, data, path):
        new_img = nib.Nifti1Image(data, self.img.affine, self.img.header)
        nib.save(new_img, path)


    #get data from seg img abt each segment
    def get_segmented_mask(self, seg_val):
        return np.where(self.img_data == seg_val, 1, 0)
    
    #
    def get_segmented_data(self, mask):
        return np.where(mask == 0, 0, self.img_data)
    
    def classify_tissue(self):
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, self.PVE = hmrf.classify(self.img_data, nclass, beta, max_iter=100) #FIX MAX ITER
        self.csf_mask = np.where((np.abs((np.maximum(np.maximum(self.PVE[:, :, :, 0], self.PVE[:, :, :, 1]), self.PVE[:, :, :, 2]) - self.PVE[:, :, :, 0])) < 0.00001), 1, 0)
        self.gray_mask = np.where((np.abs((np.maximum(np.maximum(self.PVE[:, :, :, 0], self.PVE[:, :, :, 1]), self.PVE[:, :, :, 2]) - self.PVE[:, :, :, 1])) < 0.00001), 1, 0)
        self.white_mask = np.where((np.abs((np.maximum(np.maximum(self.PVE[:, :, :, 0], self.PVE[:, :, :, 1]), self.PVE[:, :, :, 2]) - self.PVE[:, :, :, 2])) < 0.00001), 1, 0)

    def compute_volume(self, data):
        mask = np.where(data >= 0.1, 1, 0)
        return mask.sum()

dataset_ogs = list(sorted(glob.glob('/Users/yookip/ACSEF/wholeBrainSeg_Large_UNEST_segmentation/dataset/images/*.nii.gz')))
dataset_trans = list(sorted(glob.glob('/Users/yookip/ACSEF/wholeBrainSeg_Large_UNEST_segmentation/eval/*.nii.gz')))
csv_file = open('/Users/yookip/ACSEF/wholeBrainSeg_Large_UNEST_segmentation/data_points.csv', 'a')

for og_img_path in dataset_ogs:
    seg_img_path = og_img_path.replace('/dataset/images', '/eval').replace('.nii.gz', '_trans.nii.gz')
    print(og_img_path)
    print(seg_img_path)
    og_img = niiimg(img_path=og_img_path)
    seg_img = niiimg(img_path=seg_img_path)
    og_img.classify_tissue()
    out_string = ''
    for i in range (1, 133):
        parti_mask = seg_img.get_segmented_mask(i)
        parti_data = og_img.get_segmented_data(parti_mask)
        parti_img = niiimg(data=parti_data)
        #og_img.save_nii(parti_data, f'/Users/yookip/ACSEF/test/part_{i}.nii')
        
        parti_img.csf_mask = np.where((og_img.csf_mask != 0) & (parti_mask != 0), 1, 0)
        parti_img.gray_mask = np.where((og_img.gray_mask != 0) & (parti_mask != 0), 1, 0)
        parti_img.white_mask = np.where((og_img.white_mask != 0) & (parti_mask != 0), 1, 0)

        csf_parti_vol = parti_img.compute_volume(parti_img.csf_mask)
        gray_parti_vol = parti_img.compute_volume(parti_img.gray_mask)
        white_parti_vol = parti_img.compute_volume(parti_img.white_mask)
        parti_vol = parti_img.compute_volume(parti_mask)

        if csf_parti_vol+gray_parti_vol+white_parti_vol != parti_vol:
            print(f'Error: {csf_parti_vol+gray_parti_vol+white_parti_vol} != {parti_vol}')
            

        #out_string += f'{i},{segment_names[i]},{parti_vol},{csf_parti_vol},{gray_parti_vol},{white_parti_vol},'
        out_string += f'{parti_vol},{gray_parti_vol},{white_parti_vol},{csf_parti_vol},'
        print(f'working on {og_img_path} part {i} {segment_names[i]} {parti_vol} {csf_parti_vol/parti_vol:0.2f} {gray_parti_vol/parti_vol:0.2f} {white_parti_vol/parti_vol:0.2f}') 
    
    in_name = og_img_path.split('/')[-1].replace('.nii.gz', '')
    out_class = 1 if "DL" in og_img_path else (0 if "TD" in og_img_path else 2)
    csv_file.write(f'{in_name},{out_string}{out_class}\n')
    print(f'{in_name},{out_string}{out_class}\n')
csv_file.close()
    

#og_img.classify_tissue()
#og_img.save_nii(og_img.PVE[:, :, :, 0], '/Users/yookip/ACSEF/test/csf.nii')
#og_img.save_nii(og_img.PVE[:, :, :, 1], '/Users/yookip/ACSEF/test/graymatter.nii')
#og_img.save_nii(og_img.PVE[:, :, :, 2], '/Users/yookip/ACSEF/test/whitematter.nii')