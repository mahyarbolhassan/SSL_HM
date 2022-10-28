import imgaug.augmenters as iaa
import matplotlib.pyplot as plt 

sometimes = lambda aug: iaa.Sometimes(0.2, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip
    sometimes(
        iaa.ElasticTransformation(alpha=30, sigma=5)),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Noop(),
        iaa.Affine(rotate=(-90, 90)),
        iaa.Noop(),
        iaa.GaussianBlur(sigma=(0, 3.0)),
#         iaa.Noop(),
#         iaa.Crop(px=(10, 40)),
#         iaa.Noop(),
#         iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
#         iaa.Noop(),
#         iaa.LinearContrast((0.3, 1.0), per_channel=0.6),
    ]),
    # More as you want ...
])
seq_det = seq.to_deterministic()
#Move from 0-1 float to uint8 format (needed for most imgaug operators)
X_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in train_image]
# Do augmentation
X_train_aug = seq_det.augment_images(X_train_aug)
# Back to 0-1 float range
X_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in X_train_aug]

# Move from 0-1 float to uint8 format (needed for imgaug)
y_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in train_mask_multi]
# Do augmentation
y_train_aug = seq_det.augment_images(y_train_aug)
# Make sure we only have 2 values for mask augmented
y_train_aug = [np.where(x[:,:,:] > 0, 255, 0) for x in y_train_aug]
# Back to 0-1 float range
y_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in y_train_aug]
X_train_aug=np.array(X_train_aug)
y_train_aug=np.array(y_train_aug)

#################################################################################
#visualization of the augmented data
y = np.argmax(y_train_aug, axis=-1)
for i in range(10):
    img = train_image[i, :, :, :]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.5)
    ax = fig.add_subplot(2,2 , 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('original image')
    ax.imshow(np.reshape(img, (image_size, image_size)), cmap="gray")
    ax = fig.add_subplot(2,2,2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('augmented image')
    ax.imshow(np.reshape(X_train_aug[i], (image_size,image_size)), cmap="gray")
    ax = fig.add_subplot(2,2,3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('original mask')
    ax.imshow(np.reshape(train_mask[i,:,:,:], (image_size,image_size)), cmap="gray")
    ax = fig.add_subplot(2,2,4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('augmented mask')
    ax.imshow(np.reshape(y[i], (image_size,image_size)), cmap="gray")
