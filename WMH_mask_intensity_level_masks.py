import numpy as np
import os, sys
import nibabel as nib

datapath = ''

WMHIMask_1_pattern = datapath+'/{0}/WMHIMask_alpha0.5.nii.gz'
WMHIMask_2_pattern = datapath+'/{0}/WMHIMask_alpha0.8.nii.gz'
WMHIMask_3_pattern = datapath+'/{0}/WMHIMask_alpha1.1.nii.gz'
WMHIMask_4_pattern = datapath+'/{0}/WMHIMask_alpha1.4.nii.gz'
WMHIMask_5_pattern = datapath+'/{0}/WMHIMask_alpha1.7.nii.gz'
WMHIMask_6_pattern = datapath+'/{0}/WMHIMask_alpha2.1.nii.gz'
WMHIMask_7_pattern = datapath+'/{0}/WMHIMask_alpha2.4.nii.gz'
WMHIMask_8_pattern = datapath+'/{0}/WMHIMask_alpha2.7.nii.gz'

FLAIR_pattern = datapath+'/{0}/flair.nii.gz'
Seg_pattern = datapath+'/{0}/tissues_seg.nii.gz'

W1_pattern = datapath+'/{0}/W1.nii.gz'
W2_pattern = datapath+'/{0}/W2.nii.gz'
W3_pattern = datapath+'/{0}/W3.nii.gz'
W4_pattern = datapath+'/{0}/W4.nii.gz'
W5_pattern = datapath+'/{0}/W5.nii.gz'
W6_pattern = datapath+'/{0}/W6.nii.gz'
W7_pattern = datapath+'/{0}/W7.nii.gz'
W8_pattern = datapath+'/{0}/W8.nii.gz'

training_patients = sorted([f for f in os.listdir(datapath)])

#Reading the images
for p in training_patients:
    print 'processing {}'.format(p)

    FLAIR_data = nib.load(FLAIR_pattern.format(p))
    FLAIR = FLAIR_data.get_data()

    Seg_data = nib.load(Seg_pattern.format(p))
    Seg = Seg_data.get_data()


    WMHIMask1_data = nib.load(WMHIMask_1_pattern.format(p))
    WMHIMask1 = WMHIMask1_data.get_data()

    
    WMHIMask2_data = nib.load(WMHIMask_2_pattern.format(p))
    WMHIMask2 = WMHIMask2_data.get_data()        
    
    WMHIMask3_data = nib.load(WMHIMask_3_pattern.format(p))
    WMHIMask3 = WMHIMask3_data.get_data()        
    
    WMHIMask4_data = nib.load(WMHIMask_4_pattern.format(p))
    WMHIMask4 = WMHIMask4_data.get_data()        
    
    WMHIMask5_data = nib.load(WMHIMask_5_pattern.format(p))
    WMHIMask5 = WMHIMask5_data.get_data()        
    
    WMHIMask6_data = nib.load(WMHIMask_6_pattern.format(p))
    WMHIMask6 = WMHIMask6_data.get_data()        
    
    WMHIMask7_data = nib.load(WMHIMask_7_pattern.format(p))
    WMHIMask7 = WMHIMask7_data.get_data()        
    
    WMHIMask8_data = nib.load(WMHIMask_8_pattern.format(p))
    WMHIMask8 = WMHIMask8_data.get_data()        


    W8 = WMHIMask8
    nib.save(nib.Nifti1Image(np.int8(W8), WMHIMask8_data.affine), W8_pattern.format(p))

    W7= np.logical_xor(WMHIMask7, WMHIMask8)
    nib.save(nib.Nifti1Image(np.int8(W7), WMHIMask7_data.affine), W7_pattern.format(p))

    WMHIM7_8 = np.logical_or(WMHIMask7, WMHIMask8)
    W6= np.logical_xor(WMHIMask6, WMHIM7_8)
    nib.save(nib.Nifti1Image(np.int8(W6), WMHIMask6_data.affine), W6_pattern.format(p))

    WMHIM6_8 = np.logical_or(WMHIMask6, WMHIM7_8)
    W5= np.logical_xor(WMHIMask5, WMHIM6_8)
    nib.save(nib.Nifti1Image(np.int8(W5), WMHIMask5_data.affine), W5_pattern.format(p))

    WMHIM5_8 = np.logical_or(WMHIMask5, WMHIM6_8)
    W4= np.logical_xor(WMHIMask4, WMHIM5_8)
    nib.save(nib.Nifti1Image(np.int8(W4), WMHIMask4_data.affine), W4_pattern.format(p))

    WMHIM4_8 = np.logical_or(WMHIMask4, WMHIM5_8)
    W3= np.logical_xor(WMHIMask3, WMHIM4_8)
    nib.save(nib.Nifti1Image(np.int8(W3), WMHIMask3_data.affine), W3_pattern.format(p))

    WMHIM3_8 = np.logical_or(WMHIMask3, WMHIM4_8)
    W2= np.logical_xor(WMHIMask2, WMHIM3_8)
    nib.save(nib.Nifti1Image(np.int8(W2), WMHIMask2_data.affine), W2_pattern.format(p))

    WMHIM2_8 = np.logical_or(WMHIMask2, WMHIM3_8)
    W1= np.logical_xor(WMHIMask1, WMHIM2_8)
    nib.save(nib.Nifti1Image(np.int8(W1), WMHIMask1_data.affine), W1_pattern.format(p))


