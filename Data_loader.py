import numpy as np
# importing image data from vendor A, vendor B, and vendor C seperately
imgs_A = np.load('imgs_A.npy') 
msks_A = np.load('msks_A.npy') 
imgs_B = np.load('imgs_B.npy')
msks_B = np.load('msks_B.npy')
img_C = np.load('imgs_C.npy')

# the histogram matched data
hm_images = np.load('hist_match_images.npy')
hm_masks = np.load('hist_match_masks.npy')

images = np.concatenate((imgs_A, imgs_B, hm_images), axis=0)
masks = np.concatenate((msks_A, msks_B, hm_masks), axis=0)

print(images.shape, masks.shape)
