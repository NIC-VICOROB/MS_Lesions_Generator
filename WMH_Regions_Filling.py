import nibabel as nib
import os
import numpy as np
from scipy.ndimage import * 
# --------------------------------------------------
# parameters
datapath = ''

Img_pattern = datapath + '/{0}/{1}_normalized.nii.gz'
Img_filled_pattern = datapath + '/{0}/{1}_normalized_filled_WMHIM.nii.gz'
# --------------------------------------------------
scans = os.listdir(datapath)
scans.sort()
# --------------------------------------------------

for SCAN in scans:
    print ('Scan no. {}'.format(SCAN))   

    #load the lesion mask
    LES_data = nib.load(os.path.join(datapath, SCAN, 'WMHIMask_alpha0.5.nii.gz'))
    LES = LES_data.get_data() > 0

    tissues_seg_data = nib.load(os.path.join(datapath, SCAN, 'tissues_seg.nii.gz'))
    tissues_seg = tissues_seg_data.get_data()
    
    WM = tissues_seg == 3                               
    GM = tissues_seg == 2
    CSF = tissues_seg == 1                         

    #for modality in ['t1','t2','pd','flair']:    
    for modality in ['t1','flair']:
        print ('Modality. {}'.format(modality))   
        # load data
        img_data = nib.load(Img_pattern.format(SCAN, modality))
        img = img_data.get_data()
                
        img_filled = np.zeros_like(img)
        
        for s in range(img.shape[2]):            
            img_slice=img[:, :, s]            
            wm_slice=WM[:, :, s]
            les_slice=LES[:, :, s]
            
            if np.sum(wm_slice == 1)==0 or np.sum(les_slice == 1)==0 :
                img_filled[:,:,s]=img_slice
                continue
                       
            blobs, num_features = measurements.label(les_slice)    
            labels = filter(bool, np.unique(blobs))                               
            #print ('{} lesions in this slice'.format(num_features))   
            for l in labels:                
                num_lesionsvoxels = np.sum(blobs==l)
                #if num_lesionsvoxels<=10: continue
                lesion_mask = (blobs==l)
                
                lesion_mask_D10 = binary_dilation(lesion_mask, iterations=2)      
                lesion_mask_D20 = binary_dilation(lesion_mask, iterations=4)      
                
                #lesion_mask_Diff = lesion_mask_D10 - lesion_mask
                lesion_mask_Diff = np.logical_xor(lesion_mask_D10, lesion_mask)
                lesion_mask_Diff[wm_slice==0]=0
                lesion_mask_D20[wm_slice==0]=0
                
                if np.sum(lesion_mask_Diff)==0: continue 
                
                img_patch=img_slice[lesion_mask_Diff]
                
                mean_wm = np.mean(img_patch)
                std_wm = np.std(img_patch)                                
                                
                filled_les = np.random.normal(mean_wm, std_wm*0.5 ,num_lesionsvoxels)
                
                img_slice[blobs==l] = filled_les     
                         
                img_patch=img_slice[lesion_mask_D20]
                img_patch = gaussian_filter(img_patch, 0.6, output=None)                
                img_slice[lesion_mask_D20]=img_patch
                
            img_filled[:,:,s]=img_slice
        
        nib.save(nib.Nifti1Image(img_filled, img_data.affine), Img_filled_pattern.format(SCAN,modality))

