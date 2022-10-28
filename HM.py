import matplotlib.pyplot as plt
from numpy import random
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import numpy as np

# Histogram Matching

images = np.load('images.npy') 
masks = np.load('masks.npy') 
img_C = np.load('imgs_C.npy')

matched = match_histograms( images[100], img_C[50], multichannel=True)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(images[100], cmap="gray")
ax1.set_title('Source')
ax2.imshow(img_C[50], cmap="gray")
ax2.set_title('Reference')
ax3.imshow(matched, cmap="gray")
ax3.set_title('Matched')

plt.tight_layout()
plt.show()

hist_match =[]
for im in range(800):
    x = random.randint(img_C.shape[0]-1)
    matched = match_histograms( images[im], img_C[x], multichannel=True)
    hist_match.append(matched)
