import numpy as np
import os, sys
import nibabel as nib

#datapath='/home/mostafasalem/mic.hdd/mostafa/datasets/LesionsGenDataset_5/training'
#datapath='/home/mostafasalem/mic.hdd/mostafa/datasets/LesionsGenDataset_5/testing_matchedBasal/Patient'
datapath = '/home/mostafa/micServer/datasets/MSLesionsGen_4Modality'
basal_img_pattern = datapath+'/{0}/flair.nii.gz'
basal_seg_pattern = datapath+'/{0}/tissues_seg_corrected.nii.gz'

WMH_pattern = datapath+'/{0}/WMHIMask_alpha{1}.nii.gz'
                               
training_patients = sorted([f for f in os.listdir(datapath)])    


#Reading the images
for p in training_patients:    
    print 'processing {}'.format(p)
    
    basal_data = nib.load(basal_img_pattern.format(p))
    basalImg = basal_data.get_data()        
    
    basalSeg_data = nib.load(basal_seg_pattern.format(p))
    basalSeg = basalSeg_data.get_data()        
    
    for alpha in [0.5 ,0.8 ,1.1 , 1.4, 1.7, 2.1, 2.4, 2.7]:
        basal_gm_voxels = basalImg[basalSeg==2]
        basal_threshold= np.mean(basal_gm_voxels) + alpha*np.std(basal_gm_voxels)       
        basal_WMH =  basalImg > basal_threshold    
        nib.save(nib.Nifti1Image(np.int8(basal_WMH), basal_data.affine), WMH_pattern.format(p,alpha))

